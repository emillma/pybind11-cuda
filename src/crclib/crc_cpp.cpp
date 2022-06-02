#include <array>
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