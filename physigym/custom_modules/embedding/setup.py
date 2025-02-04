#####
# title: setup.py
#
# languag: setuptools
# date: 2024-02-05
# license: BSD-3-Clause
# author: Elmar Bucher
#
# description:
#   Building a setuptools based python C++ extension module.
# https://setuptools.pypa.io/en/latest/userguide/ext_modules.html
# https://elmjag.github.io/setuptools.html
#####


import os
import platform
from setuptools import Extension, setup
import sys

# extract the version number from the VERSION.txt file
exec(open('./VERSION.txt').read())


# set compile and linker arguments
ls_extra_compile_args=[  # straight outta PhysiCell Makefile
    "-march=native",  # ARCH
    "-O3",  # CFLAG
    "-fomit-frame-pointer",  # CFLAG
    "-mfpmath=both",  # CFLAG
    "-fopenmp",  # CFLAG
    "-m64",  # CFLAG
    "-std=c++11",  # CFLAG
    #"-g",  # gdb
]

ls_extra_link_args=[  # straight outta PhysiCell Makefile
    "-lgomp",  # needed for openmp
]


# set operating system specific manuipulations
if (platform.system().lower() == 'linux'):  # linux
    pass

elif (platform.system().lower() == 'windows'):  # windows
    sys.exity(f'Error: We are sorry, physigym does not compile on the native {platform.system()} operating system. However, physigym will compile on the Windows Subsystem for Linux! Give it a try!')

elif (platform.system().lower() == 'darwin'):  # apple
    os.environ["CC"] = os.environ["PHYSICELL_CPP"]
    os.environ["CXX"] = os.environ["PHYSICELL_CPP"]
    if (platform.machine().lower() == 'arm64'):  # M chipset
        ls_extra_compile_args.pop(ls_extra_compile_args.index('-mfpmath=both'))

else:
    sys.exit(f'Error: unknowen operating system {platform.system()}!')


# off we go!
setup(
    # version
    version=__version__,

    # compiler and linker ditrectives
    ext_modules = [
        Extension(
            name = "embedding.physicell",  # as it would be imported # may include packages/namespaces separated by `.`

            # all sources are compiled into a single binary file
            sources = [  # straight outta PhysiCell Makefile
                # custom_modules_OBJECTS and components
                "physicellmodule.cpp",
                "../custom.cpp",

                # BioFVM_OBJECTS and components
                "../../BioFVM/BioFVM_agent_container.cpp",
                "../../BioFVM/BioFVM_basic_agent.cpp",  # bue 20240501: modified (reset_max_basic_agent_ID) [pull request 277]
                "../../BioFVM/BioFVM_matlab.cpp",
                "../../BioFVM/BioFVM_mesh.cpp",
                "../../BioFVM/BioFVM_microenvironment.cpp",  # bue 20240501: modified (set_microenvironment_initial_condition) [pull requested 346]
                "../../BioFVM/BioFVM_MultiCellDS.cpp",  # bue 20240509: modified (reset_BioFVM_substrates_initialized_in_dom) [pull requested 266]
                "../../BioFVM/BioFVM_solvers.cpp",
                "../../BioFVM/BioFVM_utilities.cpp",
                "../../BioFVM/BioFVM_vector.cpp",
                "../../BioFVM/pugixml.cpp",

                # PhysiCell_core_OBJECTS and components
                "../../core/PhysiCell_basic_signaling.cpp",
                "../../core/PhysiCell_cell_container.cpp",
                "../../core/PhysiCell_cell.cpp",
                "../../core/PhysiCell_constants.cpp",
                "../../core/PhysiCell_custom.cpp",
                #"core/PhysiCell_digital_cell_line.cpp",
                "../../core/PhysiCell_phenotype.cpp",
                "../../core/PhysiCell_rules.cpp",
                "../../core/PhysiCell_signal_behavior.cpp",
                "../../core/PhysiCell_standard_models.cpp",
                "../../core/PhysiCell_utilities.cpp",

                # PhysiCell_module_OBJECTS and components
                "../../modules/PhysiCell_geometry.cpp",
                "../../modules/PhysiCell_MultiCellDS.cpp",
                "../../modules/PhysiCell_pathology.cpp",
                #"modules/PhysiCell_POV.cpp",
                "../../modules/PhysiCell_pugixml.cpp",
                "../../modules/PhysiCell_settings.cpp",  # bue 20240430: modified (read_PhysiCell_config_file) [pull requested 346]
                "../../modules/PhysiCell_SVG.cpp",
                "../../modules/PhysiCell_various_outputs.cpp",

                # pugixml_OBJECTS and components
                #"pugixml.cpp",
            ],

            extra_compile_args = ls_extra_compile_args,

            extra_link_args = ls_extra_link_args,
        ),
    ],
)
