# auto-calibration

~~~
${auto-calibration_ROOT}
|-- README.md  
|-- install.bash        # package requirements
|-- run.bash            # to run the code
|-- main.py             # logging / parser
|-- configs.py          # global configs
|-- utils.py            # funcs for data io, image processing, etc.
|-- method.py           # TODO: code for automatic camera calibration
|-- sample_data         # data folder
`-- |-- image00001.jpg
    |-- ...
    |-- image00664.jpg
|-- log.json            # save results

~~~


## Environment
1. Create conda environment:
`
$conda create -n auto-calibration python=3.6
`
2. Activate environment:
`
$conda activate auto-calibration
`
3. Install required packages:
`
$bash install.bash
`
## Usage
Run code script on LINUX:
`
$bash run.bash
`
