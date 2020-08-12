#include <DynamicParamStore.h>

std::map<std::string, float> dynamicParamDB::ParameterStoreDynamic;
std::map<std::string, std::string> dynamicParamDB::ParameterFilenames;
bool dynamicParamDB::ParamFileInitialized;

dynamicParamDB::dynamicParamDB(const char *filename)
{
    ParamFileInitialized = true;
    std::ifstream file(filename);
    std::string line;

    while(std::getline(file, line))
    {
        if(line[0] == '=' || line[0] == '-' || line[0] == '#')
            continue;
        else
        {
            std::regex r("[A-Za-z0-9_]+:"), rend(":.*");
            std::smatch sm, smend;
            if(std::regex_search(line, sm, r) && std::regex_search(line, smend, rend))
            {
                std::string name = sm[0].str();
                name.erase(name.size() - 1);

                std::string value = smend[0].str();

                try
                {
                    std::stof(value.c_str() + 1);
                    ParameterStoreDynamic.insert(std::make_pair(name, std::stof(value.c_str() + 1)));
                }
                catch(std::exception &e)
                {
                    value.erase(0, 1);
                    value.erase(std::remove_if(value.begin(), value.end(), isspace), value.end());
                    ParameterFilenames.insert(std::make_pair(name, value));
                    continue;
                }

                
            }
        }
    }
}

float &dynamicParamDB::operator[](const std::string &key)
{
    if(ParameterStoreDynamic.find(key) == ParameterStoreDynamic.end())
    {
        std::cerr << std::endl << std::endl << "Variable \"" << key << "\" is not available in the dat file. Please check the file!" << std::endl;
        exit(0);
    }
    else
    {
        return ParameterStoreDynamic[key];
    }
}

std::string &dynamicParamDB::GetFilepathInfo(std::string key)
{
    if(ParameterFilenames.find(key) == ParameterFilenames.end())
    {
        std::cerr << std::endl << std::endl << "Variable \"" << key << "\" is not available in the dat file. Please check the file!" << std::endl;
        exit(0);
    }
    else
    {
        return ParameterFilenames[key];
    }
}