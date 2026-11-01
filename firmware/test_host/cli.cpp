// Mac 上跑 C++ 版 tokenizer + extract_number：每讀一行字，印出「數字 | 32 個 token id」
// 給 tools/check_cpp.py 跟 Python 版逐句比對用
#include <cstdio>
#include <cstring>
#include "../src/tokenizer.h"
#include "../src/parse.h"

int main() {
  char line[4096];
  while (fgets(line, sizeof line, stdin)) {
    int len = strcspn(line, "\n");               // 去掉行尾換行
    int32_t ids[MAX_LEN];
    tokenize(line, len, ids);
    printf("%d |", extract_number(line, len));
    for (int i = 0; i < MAX_LEN; i++) printf(" %d", ids[i]);
    printf("\n");
  }
}
