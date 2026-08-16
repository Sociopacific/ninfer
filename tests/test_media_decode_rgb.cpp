#include "media/decode/decode.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <vector>

namespace {

// Position dependent and different in every channel, so a row that slid by one
// byte or by one line shows up as a mismatch instead of blending in.
std::uint8_t red(int x, int y) { return static_cast<std::uint8_t>((x * 7 + y * 13) & 0xff); }
std::uint8_t green(int x, int y) { return static_cast<std::uint8_t>((x * 3 + y * 5 + 91) & 0xff); }
std::uint8_t blue(int x, int y) { return static_cast<std::uint8_t>(((x ^ y) * 5 + 17) & 0xff); }

void put_u16(std::vector<std::uint8_t>& out, std::uint32_t value) {
    out.push_back(static_cast<std::uint8_t>(value & 0xff));
    out.push_back(static_cast<std::uint8_t>((value >> 8) & 0xff));
}

void put_u32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    put_u16(out, value & 0xffff);
    put_u16(out, (value >> 16) & 0xffff);
}

// A hand-written 24-bit BMP keeps an encoder out of the picture -- the bytes
// that come back out are exactly the bytes put in -- while still going through
// sws_scale (BGR24 -> RGB24), which is the conversion this test is about.
std::vector<std::uint8_t> bmp(int width, int height) {
    const std::size_t stride  = (static_cast<std::size_t>(width) * 3 + 3) / 4 * 4;
    const std::size_t pixels  = stride * static_cast<std::size_t>(height);
    const std::uint32_t start = 54;
    std::vector<std::uint8_t> out;
    out.reserve(start + pixels);
    out.push_back('B');
    out.push_back('M');
    put_u32(out, static_cast<std::uint32_t>(start + pixels));
    put_u32(out, 0);
    put_u32(out, start);
    put_u32(out, 40);
    put_u32(out, static_cast<std::uint32_t>(width));
    put_u32(out, static_cast<std::uint32_t>(height));
    put_u16(out, 1);
    put_u16(out, 24);
    put_u32(out, 0);
    put_u32(out, static_cast<std::uint32_t>(pixels));
    put_u32(out, 2835);
    put_u32(out, 2835);
    put_u32(out, 0);
    put_u32(out, 0);
    out.resize(start + pixels, 0);
    for (int y = 0; y < height; ++y) {
        // BMP rows run bottom to top, and the samples are stored as BGR.
        std::uint8_t* row = out.data() + start + (static_cast<std::size_t>(height - 1 - y) * stride);
        for (int x = 0; x < width; ++x) {
            row[3 * x + 0] = blue(x, y);
            row[3 * x + 1] = green(x, y);
            row[3 * x + 2] = red(x, y);
        }
    }
    return out;
}

int check(int width, int height) {
    const std::vector<std::uint8_t> bytes = bmp(width, height);
    ninfer::media::decode::Image image;
    try {
        image = ninfer::media::decode::decode_image(std::span<const std::uint8_t>(bytes),
                                                   ninfer::media::decode::Policy{});
    } catch (const std::exception& error) {
        std::cerr << width << 'x' << height << ": decode threw " << error.what() << '\n';
        return 1;
    }
    if (image.width != width || image.height != height) {
        std::cerr << width << 'x' << height << ": decoded as " << image.width << 'x' << image.height
                  << '\n';
        return 1;
    }
    const std::size_t expected = static_cast<std::size_t>(width) * height * 3;
    if (image.rgb.size() != expected) {
        std::cerr << width << 'x' << height << ": expected " << expected << " bytes, got "
                  << image.rgb.size() << '\n';
        return 1;
    }
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t at = (static_cast<std::size_t>(y) * width + x) * 3;
            if (image.rgb[at + 0] == red(x, y) && image.rgb[at + 1] == green(x, y) &&
                image.rgb[at + 2] == blue(x, y)) {
                continue;
            }
            std::cerr << width << 'x' << height << ": pixel (" << x << ',' << y << ") expected "
                      << int(red(x, y)) << ',' << int(green(x, y)) << ',' << int(blue(x, y))
                      << " got " << int(image.rgb[at + 0]) << ',' << int(image.rgb[at + 1]) << ','
                      << int(image.rgb[at + 2]) << '\n';
            return 1;
        }
    }
    return 0;
}

} // namespace

// swscale finishes an output row with a whole SIMD store, so the conversion
// buffer is allocated with padded lines and the rows are compacted afterwards.
// The compaction moves every row but the first, which is exactly the kind of
// arithmetic that stays correct on the sizes one happens to try and goes wrong
// one row over. Walk the widths instead: width*3 takes every residue modulo 32
// as the width walks 32 steps, so this covers both the rows that were always a
// round number of registers and the ones that were not.
//
// What this does NOT cover is the overrun itself. BMP arrives as BGR24 and the
// swap to RGB24 stays inside the row; only the yuv2rgb path -- what a JPEG or a
// video frame goes through -- writes past it. That one is invisible from out
// here because it lands beyond the buffer rather than in it, and ASan cannot
// see it either: libswscale is not instrumented, and its redzones absorb the
// write that kills a stock allocator. Valgrind does see it, so to check the
// padding rather than the compaction, decode a JPEG whose width*3 is not a
// multiple of 32 under valgrind and expect no "Invalid write".
int main() {
    int failures = 0;
    for (int width = 1; width <= 40; ++width) {
        for (const int height : {1, 2, 5}) { failures += check(width, height); }
    }
    // The shape that used to kill the server: 778 * 3 = 2334, and 2334 % 32 = 30.
    failures += check(778, 17);
    failures += check(778, 368);
    if (failures != 0) {
        std::cerr << failures << " media decode checks failed\n";
        return 1;
    }
    return 0;
}
