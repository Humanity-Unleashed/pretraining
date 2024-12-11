"""
for a benchmarking test, save output + info related to file

use only basic data types like pd.DataFrame, dicts etc. rather than saving binaries/class instances via something like .pkl (like I do now..) making them dependent on code version.

e.g.
{
        model_name : "",
        responses : "",
        config : {
            temperature : 1,
            ...
            },
        data_filepath : "",
        task : "",
        context : "",
        logs : "",
        package_version : "",
        etc.
}

"""
