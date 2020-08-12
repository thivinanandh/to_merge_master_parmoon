#ifndef __DYNAMIC_PARAMETERS_STORE_CLASS__
#define __DYNAMIC_PARAMETERS_STORE_CLASS__

#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <regex>
#include <exception>
#include <map>
#include <string>
#include <algorithm>

class dynamicParamDB
{
    static bool ParamFileInitialized;

public:
    static std::map<std::string, float> ParameterStoreDynamic;
    static std::map<std::string, std::string> ParameterFilenames;

    dynamicParamDB()
    {
        if(!ParamFileInitialized){
            std::cerr << "No parameter file initialized" << std::endl
                      << std::endl;
            exit(0);
        }
    }
        
    dynamicParamDB(const char *);

    float &operator[](const std::string&);

    std::string &GetFilepathInfo(std::string);
};
#endif