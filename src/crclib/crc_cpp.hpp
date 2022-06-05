#pragma once

unsigned get_crc(const unsigned char *message, int len, unsigned int crc = 0);

unsigned get_crc_lookup(const unsigned char *message, int len,
                        unsigned int crc = 0);

unsigned get_crc_lookup_parallel(const unsigned char *message, int len,
                                 const unsigned *table, unsigned int crc = 0);

unsigned join_crc_from_lookup(unsigned crc1, unsigned crc2,
                              const unsigned *join_table);