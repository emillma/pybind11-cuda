bool check_message(py::array_t<unsigned char> vec) {
    py::buffer_info buf = vec.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("wrong dim");
    }
    int len = buf.shape[0];
    unsigned char *ptr = static_cast<unsigned char *>(buf.ptr);
    unsigned int crc = get_crc(ptr, len - 4);
    unsigned int check = __builtin_bswap32(*reinterpret_cast<unsigned int *>(&ptr[len - 4]));
    return (check == crc);
}