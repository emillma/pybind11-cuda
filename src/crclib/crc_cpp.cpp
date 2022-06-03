#include <array>
#include <vector>
#define poly 0x04c11db7

unsigned get_crc(unsigned char *message, int len, unsigned int crc = 0xffffffff) {
    for (int i = 0; i < len; i++) {
        crc = crc ^ message[i];
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

unsigned get_crc_lookup(unsigned char *message, int len, unsigned int crc = 0xffffffff) {
    std::array<unsigned, 256> table = get_lookup();
    for (int i = 0; i < len; i++) {
        unsigned char val = message[i];
        crc = (0xffffff & crc) << 8 ^ table[val ^ static_cast<unsigned char>(crc >> 24)];
    }
    return crc;
}

unsigned get_crc_parallel(unsigned char *message, int len, int splits, unsigned int crc = 0xffffffff) {
    std::vector<unsigned> tmp;
    tmp.resize(splits);
    int step = len / splits;

#pragma omp parallel for num_threads(8)
    for (int i = 0; i < splits; i++) {
        tmp[i] = get_crc(&message[step * i], len / splits, 0x00000000);
    }
    for (int i = 0; i < splits; i++) {
        crc = crc ^ tmp[i];
    }
    return crc;
}

unsigned get_crc_lookup_parallel(unsigned char *message, int len, int splits, unsigned int crc = 0xffffffff) {
    std::vector<unsigned> tmp;
    tmp.resize(splits);
    int step = len / splits;

#pragma omp parallel for num_threads(8)
    for (int i = 0; i < splits; i++) {
        tmp[i] = get_crc_lookup(&message[step * i], len / splits, 0x00000000);
    }
    for (int i = 0; i < splits; i++) {
        crc = crc ^ tmp[i];
    }
    return crc;
}