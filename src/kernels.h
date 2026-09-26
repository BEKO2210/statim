#pragma once

namespace statim {
void geglu_rows(float* dst, const float* src, long rows, long ff, long row_begin, long row_end);
}
