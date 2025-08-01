#####
# title: PhyiCell/Makefile
#
# language: make
# date: 2015-2024
# license: BSD-3-Clause
# author: Paul Macklin
# modification: Elmar Bucher, Alexandre Bertin
# original source code: https://github.com/MathCancer/PhysiCell
# modified source code: https://github.com/elmbeech/physicellembedding
#####


VERSION := $(shell grep . VERSION.txt | cut -f1 -d:)
PROGRAM_NAME := project

CC := g++
ifdef PHYSICELL_CPP
	CC := $(PHYSICELL_CPP)
endif

ifndef STATIC_OPENMP
	STATIC_OPENMP = -fopenmp
endif

ARCH := native

CFLAGS := -march=$(ARCH) -O3 -fomit-frame-pointer -mfpmath=both -fopenmp -m64 -std=c++11

ifeq ($(OS),Windows_NT)
	UNAME_S := Windows_NT
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Darwin)
		UNAME_P := $(shell uname -p)
		var := $(shell which $(CC) | xargs file)
		ifeq ($(lastword $(var)),arm64)
			CFLAGS := -march=$(ARCH) -O3 -fomit-frame-pointer -fopenmp -m64 -std=c++11
		endif
	endif
endif


# compile the project

all: main.cpp
	pip3 install --force-reinstall custom_modules/extending
	pip3 install --force-reinstall --no-deps --editable custom_modules/physigym
	make python-instruction

python-instruction:
	@echo ''
	@echo 'open a Python shell and try:'
	@echo ''
	@echo 'import gymnasium'
	@echo 'import physigym'
	@echo ''
	@echo 'gymnasium.envs.pprint_registry()'
	@echo ''
	@echo 'env = gymnasium.make("physigym/ModelPhysiCellEnv")'
	@echo ''
	@echo 'help(physigym.envs.ModelPhysiCellEnv)'
	@echo ''
	@echo 'env.reset()'
	@echo 'env.step(action={})'
	@echo 'env.close()'
	@echo ''
	@echo 'exit()'
	@echo ''

install-requirement:
	pip3 install --force-reinstall --editable custom_modules/physigym


### BEGIN make classic related compiler infromation
### ACHTUNG: for physicell embedding python_with_physicell this content was ported to custom_modules/extending/setup.py!

CFLAGS_LINK := $(shell echo $(CFLAGS) | sed -e "s/-fopenmp//g")
COMPILE_COMMAND := $(CC) $(CFLAGS) $(EXTRA_FLAGS)
LINK_COMMAND := $(CC) $(CFLAGS_LINK) $(EXTRA_FLAGS)

BioFVM_OBJECTS := BioFVM_vector.o BioFVM_mesh.o BioFVM_microenvironment.o \
BioFVM_solvers.o BioFVM_matlab.o BioFVM_utilities.o BioFVM_basic_agent.o \
BioFVM_MultiCellDS.o BioFVM_agent_container.o

PhysiCell_core_OBJECTS := PhysiCell_phenotype.o PhysiCell_cell_container.o \
PhysiCell_standard_models.o PhysiCell_cell.o PhysiCell_custom.o \
PhysiCell_utilities.o PhysiCell_constants.o PhysiCell_basic_signaling.o \
PhysiCell_signal_behavior.o PhysiCell_rules.o

PhysiCell_module_OBJECTS := PhysiCell_SVG.o PhysiCell_pathology.o \
PhysiCell_MultiCellDS.o PhysiCell_various_outputs.o PhysiCell_pugixml.o \
PhysiCell_settings.o PhysiCell_geometry.o

PhysiCell_custom_module_OBJECTS := custom.o

pugixml_OBJECTS := pugixml.o

PhysiCell_OBJECTS := $(BioFVM_OBJECTS) $(pugixml_OBJECTS) $(PhysiCell_core_OBJECTS) $(PhysiCell_module_OBJECTS)

ALL_OBJECTS := $(PhysiCell_OBJECTS) $(PhysiCell_custom_module_OBJECTS)

# compile the classic project
classic: main.cpp $(ALL_OBJECTS)
	$(COMPILE_COMMAND) -o $(PROGRAM_NAME) $(ALL_OBJECTS) main.cpp
	make name

static: main.cpp $(ALL_OBJECTS) $(MaBoSS)
	$(LINK_COMMAND) $(INC) -o $(PROGRAM_NAME) $(ALL_OBJECTS) main.cpp $(LIB) -static-libgcc -static-libstdc++ $(STATIC_OPENMP)

name:
	@echo ""
	@echo "Executable name is" $(PROGRAM_NAME)
	@echo ""

# PhysiCell core components
PhysiCell_phenotype.o: ./core/PhysiCell_phenotype.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_phenotype.cpp

PhysiCell_digital_cell_line.o: ./core/PhysiCell_digital_cell_line.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_digital_cell_line.cpp

PhysiCell_cell.o: ./core/PhysiCell_cell.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_cell.cpp

PhysiCell_cell_container.o: ./core/PhysiCell_cell_container.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_cell_container.cpp

PhysiCell_standard_models.o: ./core/PhysiCell_standard_models.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_standard_models.cpp

PhysiCell_utilities.o: ./core/PhysiCell_utilities.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_utilities.cpp

PhysiCell_custom.o: ./core/PhysiCell_custom.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_custom.cpp

PhysiCell_constants.o: ./core/PhysiCell_constants.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_constants.cpp

PhysiCell_signal_behavior.o: ./core/PhysiCell_signal_behavior.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_signal_behavior.cpp

PhysiCell_rules.o: ./core/PhysiCell_rules.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_rules.cpp

# BioFVM core components (needed by PhysiCell)
BioFVM_vector.o: ./BioFVM/BioFVM_vector.cpp
	$(COMPILE_COMMAND) -c ./BioFVM/BioFVM_vector.cpp

BioFVM_agent_container.o: ./BioFVM/BioFVM_agent_container.cpp
	$(COMPILE_COMMAND) -c ./BioFVM/BioFVM_agent_container.cpp

BioFVM_mesh.o: ./BioFVM/BioFVM_mesh.cpp
	$(COMPILE_COMMAND) -c ./BioFVM/BioFVM_mesh.cpp

BioFVM_microenvironment.o: ./BioFVM/BioFVM_microenvironment.cpp
	$(COMPILE_COMMAND) -c ./BioFVM/BioFVM_microenvironment.cpp

BioFVM_solvers.o: ./BioFVM/BioFVM_solvers.cpp
	$(COMPILE_COMMAND) -c ./BioFVM/BioFVM_solvers.cpp

BioFVM_utilities.o: ./BioFVM/BioFVM_utilities.cpp
	$(COMPILE_COMMAND) -c ./BioFVM/BioFVM_utilities.cpp

BioFVM_basic_agent.o: ./BioFVM/BioFVM_basic_agent.cpp
	$(COMPILE_COMMAND) -c ./BioFVM/BioFVM_basic_agent.cpp

BioFVM_matlab.o: ./BioFVM/BioFVM_matlab.cpp
	$(COMPILE_COMMAND) -c ./BioFVM/BioFVM_matlab.cpp

BioFVM_MultiCellDS.o: ./BioFVM/BioFVM_MultiCellDS.cpp
	$(COMPILE_COMMAND) -c ./BioFVM/BioFVM_MultiCellDS.cpp

pugixml.o: ./BioFVM/pugixml.cpp
	$(COMPILE_COMMAND) -c ./BioFVM/pugixml.cpp

# standard PhysiCell modules
PhysiCell_SVG.o: ./modules/PhysiCell_SVG.cpp
	$(COMPILE_COMMAND) -c ./modules/PhysiCell_SVG.cpp

PhysiCell_pathology.o: ./modules/PhysiCell_pathology.cpp
	$(COMPILE_COMMAND) -c ./modules/PhysiCell_pathology.cpp

PhysiCell_MultiCellDS.o: ./modules/PhysiCell_MultiCellDS.cpp
	$(COMPILE_COMMAND) -c ./modules/PhysiCell_MultiCellDS.cpp

PhysiCell_various_outputs.o: ./modules/PhysiCell_various_outputs.cpp
	$(COMPILE_COMMAND) -c ./modules/PhysiCell_various_outputs.cpp

PhysiCell_pugixml.o: ./modules/PhysiCell_pugixml.cpp
	$(COMPILE_COMMAND) -c ./modules/PhysiCell_pugixml.cpp

PhysiCell_settings.o: ./modules/PhysiCell_settings.cpp
	$(COMPILE_COMMAND) -c ./modules/PhysiCell_settings.cpp

PhysiCell_basic_signaling.o: ./core/PhysiCell_basic_signaling.cpp
	$(COMPILE_COMMAND) -c ./core/PhysiCell_basic_signaling.cpp

PhysiCell_geometry.o: ./modules/PhysiCell_geometry.cpp
	$(COMPILE_COMMAND) -c ./modules/PhysiCell_geometry.cpp

# user-defined PhysiCell modules
custom.o: ./custom_modules/custom.cpp
	$(COMPILE_COMMAND) -c ./custom_modules/custom.cpp

### ACHTUNG: for physicell embedding python_with_physicell this content was ported to custom_modules/extending/setup.py!
### END: make classic related compiler infromation



# sample projects

list-sample-projects:
	@echo "sample projects::"
	@cd ./sample_projects && ls -dt1 * | grep . | sed 's!Makefile-default!!'

list-sample-projects-intracellular:
	@echo "sample projects intracellular::"
	@cd ./sample_projects_intracellular && ls -dt1 * | grep . | sed 's!empty.txt!!'


# cleanup

reset:
	make clean
	rm -f *.cpp
	cp ./sample_projects/Makefile-default Makefile
	rm -fr ./custom_modules/*
	touch ./custom_modules/empty.txt
	touch ALL_CITATIONS.txt
	rm ALL_CITATIONS.txt
	rm -fr ./config/*
	cp ./sample_projects/template/config/PhysiCell_settings.xml ./config/PhysiCell_settings.xml

clean:
	pip3 uninstall -y extending
	pip3 uninstall -y physigym
	pip3 uninstall -y UNKNOWN
	rm -fr ./custom_modules/extending/build
	rm -fr ./custom_modules/extending/custom_modules
	rm -fr ./custom_modules/extending/extending.egg-info
	rm -fr ./custom_modules/extending/UNKNOWN.egg-info
	rm -f *.o
	rm -f $(PROGRAM_NAME)*

data-cleanup:
	rm -rf ./output
	mkdir ./output
	touch ./output/empty.txt


# archival

checkpoint:
	zip -r $$(date +%b_%d_%Y_%H%M).zip Makefile *.cpp *.h config/* custom_modules/*

zip:
	zip -r latest.zip Makefile* *.cpp *.h BioFVM/* config/* core/* custom_modules/* matlab/* modules/* sample_projects/*
	cp latest.zip $$(date +%b_%d_%Y_%H%M).zip
	cp latest.zip VERSION_$(VERSION).zip
	mv *.zip archives/

tar:
	tar --ignore-failed-read -czf latest.tar Makefile* *.cpp *.h BioFVM/* config/* core/* custom_modules/* matlab/* modules/* sample_projects/*
	cp latest.tar $$(date +%b_%d_%Y_%H%M).tar
	cp latest.tar VERSION_$(VERSION).tar
	mv *.tar archives/

unzip:
	cp ./archives/latest.zip .
	unzip latest.zip

untar:
	cp ./archives/latest.tar .
	tar -xzf latest.tar


# easier animation

FRAMERATE := 24
OUTPUT := output/episode00000000

jpeg:
	@magick identify -format "%h" $(OUTPUT)/initial.svg > __H.txt
	@magick identify -format "%w" $(OUTPUT)/initial.svg > __W.txt
	@expr 2 \* \( $$(grep . __H.txt) / 2 \) > __H1.txt
	@expr 2 \* \( $$(grep . __W.txt) / 2 \) > __W1.txt
	@echo "$$(grep . __W1.txt)!x$$(grep . __H1.txt)!" > __resize.txt
	@magick mogrify -format jpg -resize $$(grep . __resize.txt) $(OUTPUT)/s*.svg
	rm -f __H*.txt __W*.txt __resize.txt

gif:
	magick convert $(OUTPUT)/s*.svg $(OUTPUT)/out.gif

movie:
	ffmpeg -r $(FRAMERATE) -f image2 -i $(OUTPUT)/snapshot%08d.jpg -vcodec libx264 -pix_fmt yuv420p -strict -2 -tune animation -crf 15 -acodec none $(OUTPUT)/out.mp4


# upgrade rules

SOURCE := PhysiCell_upgrade.zip
get-upgrade:
	@echo $$(curl https://raw.githubusercontent.com/MathCancer/PhysiCell/master/VERSION.txt) > VER.txt
	@echo https://github.com/MathCancer/PhysiCell/releases/download/$$(grep . VER.txt)/PhysiCell_V.$$(grep . VER.txt).zip > DL_FILE.txt
	rm -f VER.txt
	$$(curl -L $$(grep . DL_FILE.txt) --output PhysiCell_upgrade.zip)
	rm -f DL_FILE.txt

PhysiCell_upgrade.zip:
	make get-upgrade

upgrade: $(SOURCE)
	unzip $(SOURCE) PhysiCell/VERSION.txt
	mv -f PhysiCell/VERSION.txt .
	unzip $(SOURCE) PhysiCell/core/*
	cp -r PhysiCell/core/* core
	unzip $(SOURCE) PhysiCell/modules/*
	cp -r PhysiCell/modules/* modules
	unzip $(SOURCE) PhysiCell/sample_projects/*
	cp -r PhysiCell/sample_projects/* sample_projects
	unzip $(SOURCE) PhysiCell/BioFVM/*
	cp -r PhysiCell/BioFVM/* BioFVM
	unzip $(SOURCE) PhysiCell/documentation/User_Guide.pdf
	mv -f PhysiCell/documentation/User_Guide.pdf documentation
	rm -f -r PhysiCell
	rm -f $(SOURCE)


# user projects

# use: make command PROJ=your_project_name
# bue 20240625: don't change default to keep up with PhysiCell standard.
PROJ := my_project

save:
	echo "Saving project as $(PROJ) ... "
	mkdir -p ./user_projects
	mkdir -p ./user_projects/$(PROJ)
	mkdir -p ./user_projects/$(PROJ)/custom_modules
	mkdir -p ./user_projects/$(PROJ)/config
	cp main.cpp ./user_projects/$(PROJ)
	cp Makefile ./user_projects/$(PROJ)
	cp VERSION.txt ./user_projects/$(PROJ)
	cp -r ./config/* ./user_projects/$(PROJ)/config
	rm -fr ./custom_modules/extending/build
	rm -fr ./custom_modules/extending/custom_modules
	rm -fr ./custom_modules/extending/extending.egg-info
	rm -fr ./custom_modules/extending/UNKNOWN.egg-info
	rm -fr ./custom_modules/physigym/physigym/__pycache__
	rm -fr ./custom_modules/physigym/physigym/envs/__pycache__
	cp -r ./custom_modules/* ./user_projects/$(PROJ)/custom_modules

load:
	echo "Loading project from $(PROJ) ... "
	cp ./user_projects/$(PROJ)/main.cpp .
	cp ./user_projects/$(PROJ)/Makefile .
	cp -r ./user_projects/$(PROJ)/config/* ./config/
	cp -r ./user_projects/$(PROJ)/custom_modules/* ./custom_modules/

pack:
	@echo " "
	@echo "Preparing project $(PROJ) for sharing ... "
	@echo " "
	cd ./user_projects && zip -r $(PROJ).zip $(PROJ)
	@echo " "
	@echo "Share ./user_projects/$(PROJ).zip ... "
	@echo "Other users can unzip $(PROJ).zip in their ./user_projects, compile, and run."
	@echo " "

unpack:
	@echo " "
	@echo "Preparing shared project $(PROJ).zip for use ... "
	@echo " "
	cd ./user_projects && unzip $(PROJ).zip
	@echo " "
	@echo "Load this project via make load PROJ=$(PROJ) ... "
	@echo " "

list-user-projects:
	@echo "user projects::"
	@cd ./user_projects && ls -dt1 * | grep . | sed 's!empty.txt!!'
