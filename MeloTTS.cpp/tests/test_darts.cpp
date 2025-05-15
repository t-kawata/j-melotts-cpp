#include <iostream>
#include <vector>
#include <numeric>
#include <darts.h>
#include <string>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#ifdef _WIN32
#include <codecvt>
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#endif
#include "darts.h"
// Darts http://chasen.org/~taku/software/darts/ sample code 
// sample from https://gist.github.com/nakagami/3ca60a82337ed66590d7e70a52efe352
using namespace std;
int main(int argc, char** argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    system("chcp 65001"); //Using UTF-8 Encoding
#endif
   

  //  const Darts::DoubleArray::key_type* str[] = { "ALGOL", "ANSI", "ARCO",  "ARPA", "ARPANET", "ASCII" }; // same as char*
    std::vector<std::string>punctuation = {
    "，", "。", "！", "？", "、", "；", "：", "“", "”", "‘", "’", "（", "）", "【", "】", "《", "》", "——", "……", "·",
    ",", ".", "!", "?", ";", ":", "\"",  "\'", "/","'", "(", ")", "[", "]", "<", ">", "-", "...", ".", "\n", "\t", "\r",
    "」","「","~","～","—","】","【","$","¥ ",
    };
    std::unordered_set<std::string> sentence_splitter = {
      // "，", "。", "！", "？","；",
       ",", ".", "!", "?", ";",
    };
    std::unordered_set<std::string> spaces = {
        "\n", "\t", "\r",
    };
    std::unordered_map<std::string, char> rep_map = {
       {"：", ','},{"；", ','},{"，", ','},{"。", '.'},{"！", '!'},{"？", '?'},{"\n", '.'},{"·", ','},{"、", ','},{"...", '.'},{"$", '.'},{"“", '\''},{"”", '\''},{"‘", '\''},{"’", '\''},{"（", '\''},{"）", '\''},
       {"(", '\''},{")", '\''},{"《", '\''},{"》", '\''},{"【", '\''},{"】", '\''},{"[", '\''},{"]", '\''},{"—", '-'},{"～", '-'},{"~", '-'},{"「", '\''},{"」", '\''},{"-",'-'},{"\'",'\''}
    };

    //std::cout << punctuation[1].size()<<std::endl;
   // std::vector<std::string>chinese_punctuation = { "，", "。" };
    stable_sort(punctuation.begin(),punctuation.end());
    int n = punctuation.size();
    std::vector<const char*> keys;
    std::vector<size_t> length;
    std::vector<Darts::DoubleArray::result_type> values(n, 0);// dummy punctuation flag is 0
    for (int i = 0; i < n; ++i) {
        keys.emplace_back(punctuation[i].c_str());
        length.emplace_back(punctuation[i].size());
        if (rep_map.contains(punctuation[i])) {
            /*We are modifying the original Python logic for symbol replacement here. 
            Previously, both Chinese and English brackets were replaced with a single quote ('), 
            but this caused issues with clarity in English words, such as "(MB)". Now, we will replace them with spaces instead.*/
            int val = static_cast<int>(rep_map.at(punctuation[i]));
            if(val==int('\''))
                val = int(' ');
            values[i] = val;
            //std::cout << punctuation[i] << " "<< rep_map.at(punctuation[i])  <<" " << values[i] << std::endl;
        }
        else if (sentence_splitter.contains(punctuation[i])) {
            values[i] = static_cast<int>(punctuation[i].front());// sentence splitter flag
           
        }
        else if (spaces.contains(punctuation[i])) {
            values[i] = 3;// space flag
        }
    }

    
    //std::iota(values.begin(),values.end(),1);
    // build 
    Darts::DoubleArray da;
    da.build(keys.size(), keys.data(), length.data(), values.data());
   // da.build(21, chinese_punctuation, 0, vals.data());

    // exactMatchSearch
    //cout << da.exactMatchSearch<Darts::DoubleArray::result_type>("ALGOL") << endl;
    //cout << da.exactMatchSearch<Darts::DoubleArray::result_type>("ANSI") << endl;
    //cout << da.exactMatchSearch<Darts::DoubleArray::result_type>("ARCO") << endl;;
    //cout << da.exactMatchSearch<Darts::DoubleArray::result_type>("ARPA") << endl;;
    //cout << da.exactMatchSearch<Darts::DoubleArray::result_type>("ARPANET") << endl;;
    //cout << da.exactMatchSearch<Darts::DoubleArray::result_type>("ASCII") << endl;;
    //cout << da.exactMatchSearch<Darts::DoubleArray::result_type>("APPARE") << endl;

    // commonPrefixSearch
    Darts::DoubleArray::result_pair_type  result_pair[1024];
    size_t num = da.commonPrefixSearch("，逗号这种都不要了", result_pair, sizeof(result_pair));
    std::cout << "found:" << num << endl;
    for (size_t i = 0; i < num; ++i) {
        cout << "\tvalue:" << result_pair[i].value << " matched key length:" << result_pair[i].length << endl;
    }

    num = da.commonPrefixSearch("。逗号这种都不要了", result_pair, sizeof(result_pair));
    cout << "found:" << num << endl;

    // save to file
    da.save("punc_.dic");
    da.clear();

    // load from file and commonPrefixSearch
    Darts::DoubleArray da2;
    da2.open("punc_.dic");
    num = da2.commonPrefixSearch("。逗号这种都不要了", result_pair, sizeof(result_pair));
    cout << "found:" << num << endl;
    da2.clear();
}