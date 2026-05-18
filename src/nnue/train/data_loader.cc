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
    - pieces: each byte holds up to two pieces 0-11
        - ordered as they are on the board [a8, b8, ... g1, h1]
        - pieces[i] = pc0 | (pc1 << 4)
    - occupancy: bitboard
    - white/black king squares
    - eval: from perspective of side to move
    - best_move: (from_square << 6) | to_square
    - result: -1 loss, 0 draw, 1 win, from perspective of side to move
    - turn: 0 white, 1 black

    Could be slightly more compact, but it is good to be 32 bytes
    */
    struct DataPoint {
        uint8_t pieces[16];
        uint64_t occupancy;
        uint8_t white_king;
        uint8_t black_king;
        int16_t eval;
        uint16_t best_move;
        int8_t result;
        uint8_t turn;
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

                int offset = batch_index * NUM_FEATURES;
                float* out_white;
                float* out_black;
                if (pos.turn == 0) {
                    out_white = stm_ptr + offset;
                    out_black = opp_ptr + offset;
                } else {
                    out_white = opp_ptr + offset;
                    out_black = stm_ptr + offset;
                }

                bool white_mirror = (pos.white_king % 8) > 3;
                bool black_mirror = (pos.black_king % 8) > 3;

                uint64_t occupancy = pos.occupancy;
                int num_pieces = __builtin_popcountll(occupancy);
                for (int i = 0; i < num_pieces; i++) {
                    int idx = i / 2;
                    int white_piece = (pos.pieces[idx] >> (4 * (i % 2))) & 0xF;
                    int black_piece = white_piece - 6 + 12 * (white_piece <= 5);

                    int white_square = __builtin_ctzll(occupancy);
                    int black_square = white_square ^ 0b111000;
                    white_square ^= white_mirror * 0b111;
                    black_square ^= black_mirror * 0b111;

                    out_white[white_square * 12 + white_piece] = 1;
                    out_black[black_square * 12 + black_piece] = 1;

                    occupancy &= occupancy - 1;
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