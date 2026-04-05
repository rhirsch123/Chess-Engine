#include <torch/extension.h>
#include <ATen/Parallel.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
    /* 
    Data format:
    pos: 16 identifiers of pieces on 4 adjacent squares {a8-d8, e8-h8, a7-d7, ...}
       - pos[i] = pc0 + pc1 * 13 + pc2 * 13^2 + pc3 * 13^3
    eval: from perspective of side to move
    result: -1 loss, 0 draw, 1 win, from perspective of side to move
    turn: 0 white, 1 black
    */
    struct DataPoint {
        uint16_t pos[16];
        int16_t eval;
        int8_t result;
        int8_t turn;
    };

    struct MMapFile {
        int fd = -1;
        size_t size = 0;
        void* data = nullptr;

        MMapFile() = default;

        MMapFile(const std::string& path) { open_map(path); }

        void open_map(const std::string& path) {
            fd = ::open(path.c_str(), O_RDONLY);
            if (fd < 0) {
                throw std::runtime_error("Failed to open: " + path);
            }

            struct stat st;
            if (fstat(fd, &st) != 0) {
                ::close(fd);
                fd = -1;
                throw std::runtime_error("fstat failed: " + path);
            }

            size = static_cast<size_t>(st.st_size);
            data = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (data == MAP_FAILED) {
                ::close(fd);
                fd = -1;
                data = nullptr;
                throw std::runtime_error("mmap failed: " + path);
            }

            #if defined(MADV_SEQUENTIAL)
            ::madvise(data, size, MADV_SEQUENTIAL);
            #endif
        }

        ~MMapFile() {
            if (data && data != MAP_FAILED) ::munmap(data, size);
            if (fd >= 0) ::close(fd);
        }
    };
}



class NNUEBatchLoader {
public:

    NNUEBatchLoader(std::string data_path, uint64_t num_positions, bool pin_memory)
        : data_file(data_path), num_positions(num_positions),
          pin_memory(pin_memory), data_map(data_file) {

        const size_t data_needed = static_cast<size_t>(num_positions) * sizeof(DataPoint);
        if (data_map.size < data_needed) {
            throw std::runtime_error("More positions than in file");
        }

        data = reinterpret_cast<const DataPoint *>(data_map.data);
    }

    uint64_t size() const { 
        return num_positions;
    }

    // returns (stm_features, opp_features, eval)
    std::vector<torch::Tensor> get_batch(uint64_t start, uint64_t batch_size) const {
        static constexpr int NUM_FEATURES = 768;
        static constexpr int NUM_PIECES = 12;

        auto feature_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        auto eval_options = torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU);
        if (pin_memory) { 
            feature_options = feature_options.pinned_memory(true);
            eval_options = eval_options.pinned_memory(true);
        }

        const uint64_t end = std::min<uint64_t>(num_positions, start + batch_size);
        const int64_t B = end - start;

        auto x_stm = torch::zeros({B, NUM_FEATURES}, feature_options);
        auto x_opp = torch::zeros({B, NUM_FEATURES}, feature_options);
        auto eval = torch::empty({B}, eval_options);

        float* stm_ptr = x_stm.data_ptr<float>();
        float* opp_ptr = x_opp.data_ptr<float>();
        int16_t* eval_ptr = eval.data_ptr<int16_t>();

        // parallel over samples in the batch
        at::parallel_for(0, B, 1, [&](uint64_t low, uint64_t high) {
            for (uint64_t batch_index = low; batch_index < high; batch_index++) {
                const DataPoint pos = data[start + batch_index];
                eval_ptr[batch_index] = pos.eval;

                float* out_stm = stm_ptr + batch_index * NUM_FEATURES;
                float* out_opp = opp_ptr + batch_index * NUM_FEATURES;

                int white_king_square = 0;
                int black_king_square = 0;
                for (int i = 0; i < 16; i++) {
                    uint16_t index = pos.pos[i];
                    
                    int square = 4 * i;

                    int pc0 = index % 13;
                    index /= 13;
                    int pc1 = index % 13;
                    index /= 13;
                    int pc2 = index % 13;
                    index /= 13;
                    int pc3 = index % 13;

                    if (pc0 == 12) black_king_square = square + 0;
                    else if (pc1 == 12) black_king_square = square + 1;
                    else if (pc2 == 12) black_king_square = square + 2;
                    else if (pc3 == 12) black_king_square = square + 3;

                    if (pc0 == 6) white_king_square = square + 0;
                    else if (pc1 == 6) white_king_square = square + 1;
                    else if (pc2 == 6) white_king_square = square + 2;
                    else if (pc3 == 6) white_king_square = square + 3;
                }
                bool white_mirror = (white_king_square % 8) > 3;
                bool black_mirror = (black_king_square % 8) > 3;

                if (pos.turn == 0) {
                    for (int i = 0; i < 16; i++) {
                        uint16_t index = pos.pos[i];
                        
                        int square = 4 * i;
                        int black_square = square ^ 0b111000;

                        if (white_mirror) square ^= 0b111;
                        if (black_mirror) black_square ^= 0b111;

                        float* ft_stm = out_stm + square * 12;
                        float* ft_opp = out_opp + black_square * 12;

                        int pc0 = index % 13;
                        index /= 13;
                        int pc1 = index % 13;
                        index /= 13;
                        int pc2 = index % 13;
                        index /= 13;
                        int pc3 = index % 13;

                        if (pc0) {
                            int white_offset = white_mirror ? -NUM_PIECES * 0 : NUM_PIECES * 0;
                            int black_offset = black_mirror ? -NUM_PIECES * 0 : NUM_PIECES * 0;

                            ft_stm[white_offset + pc0 - 1] = 1;
                            int black_piece = pc0 <= 6 ? pc0 + 5 : pc0 - 7;
                            ft_opp[black_offset + black_piece] = 1;
                        }
                        if (pc1) {
                            int white_offset = white_mirror ? -NUM_PIECES * 1 : NUM_PIECES * 1;
                            int black_offset = black_mirror ? -NUM_PIECES * 1 : NUM_PIECES * 1;

                            ft_stm[white_offset + pc1 - 1] = 1;
                            int black_piece = pc1 <= 6 ? pc1 + 5 : pc1 - 7;
                            ft_opp[black_offset + black_piece] = 1;
                        }
                        if (pc2) {
                            int white_offset = white_mirror ? -NUM_PIECES * 2 : NUM_PIECES * 2;
                            int black_offset = black_mirror ? -NUM_PIECES * 2 : NUM_PIECES * 2;

                            ft_stm[white_offset + pc2 - 1] = 1;
                            int black_piece = pc2 <= 6 ? pc2 + 5 : pc2 - 7;
                            ft_opp[black_offset + black_piece] = 1;
                        }
                        if (pc3) {
                            int white_offset = white_mirror ? -NUM_PIECES * 3 : NUM_PIECES * 3;
                            int black_offset = black_mirror ? -NUM_PIECES * 3 : NUM_PIECES * 3;

                            ft_stm[white_offset + pc3 - 1] = 1;
                            int black_piece = pc3 <= 6 ? pc3 + 5 : pc3 - 7;
                            ft_opp[black_offset + black_piece] = 1;
                        }
                    }
                } else {
                    for (int i = 0; i < 16; i++) {
                        uint16_t index = pos.pos[i];
                        
                        int square = 4 * i;
                        int black_square = square ^ 0b111000;

                        if (white_mirror) square ^= 0b111;
                        if (black_mirror) black_square ^= 0b111;

                        float* ft_opp = out_opp + square * 12;
                        float* ft_stm = out_stm + black_square * 12;

                        int pc0 = index % 13;
                        index /= 13;
                        int pc1 = index % 13;
                        index /= 13;
                        int pc2 = index % 13;
                        index /= 13;
                        int pc3 = index % 13;

                        if (pc0) {
                            int white_offset = white_mirror ? -NUM_PIECES * 0 : NUM_PIECES * 0;
                            int black_offset = black_mirror ? -NUM_PIECES * 0 : NUM_PIECES * 0;

                            ft_opp[white_offset + pc0 - 1] = 1;
                            int black_piece = pc0 <= 6 ? pc0 + 5 : pc0 - 7;
                            ft_stm[black_offset + black_piece] = 1;
                        }
                        if (pc1) {
                            int white_offset = white_mirror ? -NUM_PIECES * 1 : NUM_PIECES * 1;
                            int black_offset = black_mirror ? -NUM_PIECES * 1 : NUM_PIECES * 1;

                            ft_opp[white_offset + pc1 - 1] = 1;
                            int black_piece = pc1 <= 6 ? pc1 + 5 : pc1 - 7;
                            ft_stm[black_offset + black_piece] = 1;
                        }
                        if (pc2) {
                            int white_offset = white_mirror ? -NUM_PIECES * 2 : NUM_PIECES * 2;
                            int black_offset = black_mirror ? -NUM_PIECES * 2 : NUM_PIECES * 2;

                            ft_opp[white_offset + pc2 - 1] = 1;
                            int black_piece = pc2 <= 6 ? pc2 + 5 : pc2 - 7;
                            ft_stm[black_offset + black_piece] = 1;
                        }
                        if (pc3) {
                            int white_offset = white_mirror ? -NUM_PIECES * 3 : NUM_PIECES * 3;
                            int black_offset = black_mirror ? -NUM_PIECES * 3 : NUM_PIECES * 3;

                            ft_opp[white_offset + pc3 - 1] = 1;
                            int black_piece = pc3 <= 6 ? pc3 + 5 : pc3 - 7;
                            ft_stm[black_offset + black_piece] = 1;
                        }
                    }
                }
            }
        });

        return {x_stm, x_opp, eval};
    }

private:
    std::string data_file;
    uint64_t num_positions;
    bool pin_memory;
    MMapFile data_map;
    const DataPoint* data = nullptr;
};



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<NNUEBatchLoader>(m, "NNUEBatchLoader")

    .def(
        py::init<std::string, uint64_t, bool>(),
        py::arg("data_path"),
        py::arg("num_positions"),
        py::arg("pin_memory") = false
    )

    .def("size", &NNUEBatchLoader::size)

    .def(
        "get_batch", &NNUEBatchLoader::get_batch,
        py::arg("start"),
        py::arg("batch_size")
    );
}