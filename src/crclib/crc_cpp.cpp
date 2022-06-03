#include <array>
#include <vector>
#define poly 0x04c11db7
#define parallel_chunk 1024

unsigned get_crc(const unsigned char *message, int len, unsigned int crc = 0) {
    for (int i = 0; i < len; i++) {
        crc = crc ^ static_cast<unsigned>(message[i]) << 24;
        for (int j = 0; j < 8; j++) {
            if (crc & 0x80000000) {
                crc = (crc << 1) ^ poly;
            } else {
                crc = crc << 1;
            }
        }
    }
    return crc;
}

constexpr std::array<unsigned, 256> get_lookup() {
    std::array<unsigned, 256> table{};
    for (unsigned i = 0; i < 256; i++) {
        unsigned bflip = i << 24;
        for (int j = 0; j < 8; j++) {
            if (bflip & 0x80000000) {
                bflip = (bflip << 1) ^ poly;
            } else {
                bflip = bflip << 1;
            }
        }
        table[i] = bflip;
    }
    return table;
}

unsigned get_crc_lookup(const unsigned char *message, int len, unsigned int crc = 0) {
    const std::array<unsigned, 256> table = get_lookup();
    for (int i = 0; i < len; i++) {
        unsigned char val = message[i];
        crc = (0xffffff & crc) << 8 ^ table[val ^ static_cast<unsigned char>(crc >> 24)];
    }
    return crc;
}

constexpr std::array<unsigned, 256 * 4> get_join_table(int dist) {
    std::array<unsigned, 256 * 4> table{};
    unsigned char zeros = 0;
    for (unsigned i = 0; i < 4; i++) {
        for (unsigned j = 0; j < 256; j++) {
            unsigned flip = j << (i * 8);
            for (unsigned k = 0; k < dist; k++) {
                flip = get_crc(&zeros, 1, flip);
            }
            table[i * 256 + j] = flip;
        }
    }
    return table;
}

unsigned get_crc_lookup_parallel(const unsigned char *message, int len, unsigned int crc = 0) {
    const std::array<unsigned, 256 * 4> table = get_join_table(parallel_chunk);
    // std::array<unsigned, 256 * 4> table;
    std::vector<unsigned> tmp;
    tmp.resize(len / parallel_chunk);

    // #pragma omp parallel for num_threads(8)
    for (int i = 0; i < len / parallel_chunk; i++) {
        tmp[i] = get_crc_lookup(&message[parallel_chunk * i], parallel_chunk, 0x00000000);
    }

    for (int i = 0; i < len / parallel_chunk; i++) {
        unsigned crc_tmp = tmp[i];
        for (int j = 0; j < 4; j++) {
            crc_tmp ^= table[j * 256 + (crc >> (j * 8) & 0xff)];
        }
        crc = crc_tmp;
    }
    crc = get_crc_lookup(&message[len - len % parallel_chunk], len % parallel_chunk, crc);
    return crc;
}
