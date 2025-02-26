#include <array>
#include <vector>

#define poly 0x04c11db7
#define splits 16

template <unsigned N>
unsigned char get_byte(unsigned value) {
    return reinterpret_cast<uint8_t *>(&value)[N];
}

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

unsigned get_crc_lookup(const unsigned char *message, int len,
                        unsigned int crc = 0) {
    const std::array<unsigned, 256> table = get_lookup();
    for (int i = 0; i < len; i++) {
        crc = crc << 8 ^ table[message[i] ^ crc>>24];
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

unsigned join_crc_from_lookup(unsigned crc1, unsigned crc2, const unsigned* join_table) {
    unsigned crc_tmp = crc2;
    for (int byte = 0; byte<4; byte++) {
        crc_tmp ^= join_table[byte*256+ (crc1 >> (byte*8) & 0xff)];
    }
    return crc_tmp;
}

unsigned get_crc_lookup_parallel(const unsigned char *message, int len,
                                 const unsigned *table, unsigned crc = 0) {
    std::array<unsigned, splits> tmp;
    int step = len / splits;
    int a = 0;

    #pragma omp parallel for num_threads(splits)
    for (int i = 0; i < splits; i++) {
        tmp[i] = get_crc_lookup(&message[step * i], step, 0x00000000);
    }

    for (int i = 0; i < splits; i++) {
        crc = join_crc_from_lookup(crc, tmp[i], table);
    }
    return crc;
}
