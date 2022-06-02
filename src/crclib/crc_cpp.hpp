#pragma once

unsigned int get_crc(unsigned char *message, int len, unsigned int crc = 0xffffffff);
unsigned int get_crc_lookup(unsigned char *message, int len, unsigned int crc = 0xffffffff);
