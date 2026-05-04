////////
// title: PhyiCell/custom_modules/physicellmodule.cpp
// derived from: PhyiCell/main.cpp
//
// language: C/C++
// date: 2015-2025
// license: BSD-3-Clause
// author: Alexandre Bertin, Elmar Bucher, Paul Macklin
// original source code: https://github.com/MathCancer/PhysiCell
// modified source code: https://github.com/elmbeech/physicellembedding
// modified source code: https://github.com/Dante-Berth/PhysiGym
// input: https://docs.python.org/3/extending/extending.html
//
// description:
//   for physicell embedding the content of the regular main.cpp
//   was ported to this physicellmodule.cpp file.
////////


// load Python API
// since Python may define some pre-processor definitions which affect the standard headers on some systems, you must include Python.h before any standard headers are included.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

// load standard library
#include <stdbool.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <sys/stat.h>

#include <fcntl.h>


// loade PhysiCell library
#include "../../core/PhysiCell.h"
#include "../../modules/PhysiCell_standard_modules.h"
#include "../../custom_modules/custom.h"

// load namespace
using namespace BioFVM;
using namespace PhysiCell;

// global variable
char filename[1024];
std::ofstream report_file;
//std::vector<std::string> (*cell_coloring_function)(Cell*) = my_coloring_function;  // set a pathology coloring function
//std::string (*substrate_coloring_function)(double, double, double) = paint_by_density_percentage;

// Helper functions
// exactly like in PhysiCell BUT 
void modified_delete_cell(Cell* pDeleteMe) {
    pDeleteMe->remove_all_attached_cells();
    pDeleteMe->remove_all_spring_attachments();
    pDeleteMe->remove_self_from_all_neighbors();
    pDeleteMe->internalized_substrates->assign(pDeleteMe->internalized_substrates->size() , 0.0 );  // instead of pDeleteMe->release_internalized_substrates (BioFVM causing a core dumped)

    // move last item to index and pop
    (*all_cells)[ (*all_cells).size() - 1 ]->index = pDeleteMe->index;
    (*all_cells)[pDeleteMe->index] = (*all_cells)[ (*all_cells).size() - 1 ];
    all_cells->pop_back();

    // deregister and delete
    pDeleteMe->get_container()->remove_agent(pDeleteMe);
    delete pDeleteMe;
}

// destroy all safely
// destroy all safely
void destroy_all_cells() {
    // 1. The Clever Edge: Sever all graph relationships globally first.
    // This prevents any cell from reaching into another cell's memory during deletion.
    for (int i = 0; i < all_cells->size(); ++i) {
        (*all_cells)[i]->state.neighbors.clear();
        (*all_cells)[i]->state.attached_cells.clear();
        (*all_cells)[i]->state.spring_attachments.clear();
    }

    // 2. Now safely delete them. 
    // The remove_self_from_all_neighbors() will just see empty lists and safely do nothing.
    for (int i = all_cells->size() - 1; i >= 0; --i) {
        modified_delete_cell((*all_cells)[i]);
    }
}
//

// extended Python C++ function start
static PyObject* physicell_start(PyObject *self, PyObject *args) {

    // ===== SILENCE START =====
    int stdout_fd = dup(STDOUT_FILENO);
    int stderr_fd = dup(STDERR_FILENO);

    int devnull = open("/dev/null", O_WRONLY);
    dup2(devnull, STDOUT_FILENO);
    dup2(devnull, STDERR_FILENO);
    close(devnull);
    // =========================

    // extract args take default if no args
    char *settingxml = "config/PhysiCell_settings.xml";
    int reload = false;
    if (!PyArg_ParseTuple(args, "|sp", &settingxml, &reload)) { return NULL; }

    // reset global variables
    std::cout << "(re)set global variables ..." << std::endl;
    PhysiCell_globals = PhysiCell_Globals();

    // time setup
    std::string time_units = "min";

    // densities and cell types can only be defined in the first episode
    // and have to be reloaded in all following episodes!
    if (! reload) {
        // load xml file
        std::cout << "load setting xml " << settingxml << " ..." << std::endl;
        bool XML_status = false;
        XML_status = load_PhysiCell_config_file(settingxml);
        //PhysiCell_settings.max_time = 1440 + (std::rand() % (10080 - 1440 + 1));
        //PhysiCell_settings.max_time = 1440;
        create_output_directory(PhysiCell_settings.folder);

        // OpenMP setup
        omp_set_num_threads(PhysiCell_settings.omp_num_threads);

        // setup microenviroment and mechanics voxel size and match the data structure to BioFVM
        std::cout << "set densities ..." << std::endl;
        setup_microenvironment();  // modify this in the custom code
        double mechanics_voxel_size = 30;
        Cell_Container* cell_container = create_cell_container_for_microenvironment(microenvironment, mechanics_voxel_size);

        // load cell type definition and setup tisse
        std::cout << "load cell type definition and setup tissue ..." << std::endl;
        create_cell_types();  // modify this in the custom code
        setup_tissue();  // modify this in the custom code

        // set MultiCellDS save options
        set_save_biofvm_mesh_as_matlab(true);
        set_save_biofvm_data_as_matlab(true);
        set_save_biofvm_cell_data(true);
        set_save_biofvm_cell_data_as_custom_matlab(true);

    } else {
        // load xml file
        std::cout << "load setting xml " << settingxml << " ..." << std::endl;
        bool XML_status = false;
        XML_status = read_PhysiCell_config_file(settingxml);
        if (XML_status) { PhysiCell_settings.read_from_pugixml(); }
        if (!XML_status) { exit(-1); }
        
        create_output_directory(PhysiCell_settings.folder);

        // OpenMP setup
        omp_set_num_threads(PhysiCell_settings.omp_num_threads);

        // reset cells
        std::cout << "reset cells ..." << std::endl;
        destroy_all_cells();
        
        std::cout << "reset_max_basic_agent_ID ..." << std::endl;
        BioFVM::reset_max_basic_agent_ID();

        // reset mesh0
        std::cout << "reset mesh0 ..." << std::endl;
        BioFVM::reset_BioFVM_substrates_initialized_in_dom();

        // reset microenvironment
        std::cout << "reset densities ..." << std::endl;
        set_microenvironment_initial_condition();
        microenvironment.display_information(std::cout);

        // ✅ REMOVED the create_cell_container_for_microenvironment lines!
        // The old container is still there and perfectly empty.

        // reset tissue
        std::cout << "reset tissue ..." << std::endl;
        display_cell_definitions(std::cout);
        setup_tissue();  
    }

    // copy config file to output directory
    char copy_command [1024];
    sprintf(copy_command, "cp %s %s", settingxml, PhysiCell_settings.folder.c_str());
    system(copy_command);

    // copy seeding file
    // nop

    // save initial data simulation snapshot
    sprintf(filename, "%s/initial", PhysiCell_settings.folder.c_str());
    save_PhysiCell_to_MultiCellDS_v2(filename, microenvironment, PhysiCell_globals.current_time);

    // save data simulation snapshot output00000000
    //char filename[1024];  // bue 20240130: going global
    if (PhysiCell_settings.enable_full_saves == true) {
        sprintf(filename, "%s/output%08u", PhysiCell_settings.folder.c_str(), PhysiCell_globals.full_output_index);
        save_PhysiCell_to_MultiCellDS_v2(filename, microenvironment, PhysiCell_globals.current_time);
    }

    // save initial svg cross section through z = 0 and legend
    PhysiCell_SVG_options.length_bar = 200;  // set cross section length bar to 200 microns
    std::vector<std::string> (*cell_coloring_function)(Cell*) = my_coloring_function;  // set pathology coloring function // bue 20240130: not going global
    std::string (*substrate_coloring_function)(double, double, double) = paint_by_density_percentage;   // bue 20241026: not going global

    sprintf(filename, "%s/legend.svg", PhysiCell_settings.folder.c_str());
    create_plot_legend(filename, cell_coloring_function);

    sprintf(filename, "%s/initial.svg", PhysiCell_settings.folder.c_str());
    SVG_plot(filename, microenvironment, 0.0, PhysiCell_globals.current_time, cell_coloring_function, substrate_coloring_function);

    // save svg cross section snapshot00000000
    if (PhysiCell_settings.enable_SVG_saves == true) {
        sprintf(filename, "%s/snapshot%08u.svg", PhysiCell_settings.folder.c_str(), PhysiCell_globals.SVG_output_index);
        SVG_plot(filename, microenvironment, 0.0, PhysiCell_globals.current_time, cell_coloring_function, substrate_coloring_function);
    }

    // save legacy simulation report
    //std::ofstream report_file;  // bue 20240130: going global
    if (PhysiCell_settings.enable_legacy_saves == true) {
        sprintf(filename, "%s/simulation_report.txt", PhysiCell_settings.folder.c_str());
        report_file.open(filename);  // create the data log file
        report_file << "simulated time\tnum cells\tnum division\tnum death\twall time" << std::endl;
        log_output(PhysiCell_globals.current_time, PhysiCell_globals.full_output_index, microenvironment, report_file);  // output00000000
    }

    // standard output
    display_citations();
    display_simulation_status(std::cout);  // output00000000

    // set the performance timers
    BioFVM::RUNTIME_TIC();
    BioFVM::TIC();

    // ===== RESTORE OUTPUT =====
    dup2(stdout_fd, STDOUT_FILENO);
    dup2(stderr_fd, STDERR_FILENO);
    close(stdout_fd);
    close(stderr_fd);
    // ==========================

    // going home
    return PyLong_FromLong(0);
}


// extended Python C++ function step
static PyObject* physicell_step(PyObject *self, PyObject *args) {
    // ===== SILENCE START =====
    int stdout_fd = dup(STDOUT_FILENO);
    int stderr_fd = dup(STDERR_FILENO);

    int devnull = open("/dev/null", O_WRONLY);
    dup2(devnull, STDOUT_FILENO);
    dup2(devnull, STDERR_FILENO);
    close(devnull);
    // =========================
    // main loop

    // bue 2024-02-01: simplify (no try catch)
    //try {

        // set time variables
        // achtung : begin physigym specific implementation!
        double custom_countdown = parameters.doubles("dt_gym");  // [min]
        // achtung : end physigym specific implementation!
        double phenotype_countdown = phenotype_dt;
        double mechanics_countdown = mechanics_dt;
        double mcds_countdown = PhysiCell_settings.full_save_interval;
        double svg_countdown = PhysiCell_settings.SVG_save_interval;

        // run diffusion time step paced main loop
        bool action = true;
        bool step = true;
        while (step) {
            // run microenvironment
            microenvironment.simulate_diffusion_decay(diffusion_dt);

            // run PhysiCell
            ((Cell_Container *)microenvironment.agent_container)->update_all_cells(PhysiCell_globals.current_time);

            // max time reached?
            if (PhysiCell_globals.current_time > PhysiCell_settings.max_time) {
                step = false;
            }

            // do action
            if (action) {

                // achtung : begin physigym specific implementation!
                // std::cout << "administer drug ... " << std::endl;
                add_substrate("drug_1", parameters.doubles("drug_1"));
                action = false;
                // achtung : end physigym specific implementation!

                // Put physigym related parameter, variable, and vector action mapping here!

                // parameter
                //my_function( parameters.bools("my_bool")) );
                //my_function( parameters.ints("my_int")) );
                //my_function( parameters.doubles("my_float") );
                //my_function( parameters.strings("my_str") );

                // custom variable
                //std::string my_variable = "my_variable";
                //for (Cell* pCell : (*all_cells)) {
                //    my_function( pCell->custom_data[my_variable] );
                //}

                // custom vector
                //std::string my_vector = "my_vector";
                //for (Cell* pCell : (*all_cells)) {
                //    int vectindex = pCell->custom_data.find_vector_variable_index(my_vector);
                //    if (vectindex > -1) {
                //        my_function( pCell->custom_data.vector_variables[vectindex].value );
                //    } else {
                //        char error[1024];
                //        snprintf(error, sizeof(error), "Error: unknown custom_data vector! %s", my_vector);
                //        PyErr_SetString(PyExc_KeyError, error);
                //        return NULL;
                //    }
                //}

            }

            // do observation
            // on dt_gym time step
            if (custom_countdown < 0.5 * diffusion_dt) {

                // achtung : begin physigym specific implementation!
                custom_countdown += parameters.doubles("dt_gym");  // [min]
                // std::cout << "processing gym time step observation block ... " << std::endl;
                parameters.doubles("time") = PhysiCell_globals.current_time;
                action = true;
                step = false;
                // achtung : end physigym specific implementation!

                // Put physigym related parameter, variable, and vector observation mapping here!

                // parameter
                //parameters.bools("my_bool") = value;
                //parameters.ints("my_int") = value;
                //parameters.doubles("my_float") = value;
                //parameters.strings("my_str") = value;

                // custom variable
                //std::string my_variable = "my_variable";
                //for (Cell* pCell : (*all_cells)) {
                //    pCell->custom_data[my_variable] = value;
                //}

                // custom vector
                //std::string my_vector = "my_vector";
                //for (Cell* pCell : (*all_cells)) {
                //    int vectindex = pCell->custom_data.find_vector_variable_index(my_vector);
                //    if (vectindex > -1) {
                //        pCell->custom_data.vector_variables[vectindex].value = value;
                //    } else {
                //        char error[1024];
                //        snprintf(error, sizeof(error), "Error: unknown custom_data vector! %s", my_vector);
                //        PyErr_SetString(PyExc_KeyError, error);
                //        return NULL;
                //    }
                //}
            }

            // on phenotype time step
            if (phenotype_countdown < 0.5 * diffusion_dt) {
                phenotype_countdown += phenotype_dt;

                // Put phenotype time scale code here!
                //std::cout << "processing phenotype time step observation block ... " << std::endl;
                //step = false;
            }

            // on mechanics time step
            if (mechanics_countdown < 0.5 * diffusion_dt) {
                mechanics_countdown += mechanics_dt;

                // Put mechanics time scale code here!
                //std::cout << "processing mechanic time step observation block ... " << std::endl;
                //step = false;
            }

            // on diffusion time step

            // Put diffusion time scale code here!
            //std::cout << "processing diffusion time step observation block ... " << std::endl << std::endl;
            //step = false;


            // update time
            custom_countdown -= diffusion_dt;
            phenotype_countdown -= diffusion_dt;
            mechanics_countdown -= diffusion_dt;
            mcds_countdown -= diffusion_dt;
            svg_countdown -= diffusion_dt;
            PhysiCell_globals.current_time += diffusion_dt;

            // save data if it's time.
            if (mcds_countdown < 0.5 * diffusion_dt) {
                mcds_countdown += PhysiCell_settings.full_save_interval;
                PhysiCell_globals.full_output_index++;

                display_simulation_status(std::cout);

                // save data simulation snapshot
                if (PhysiCell_settings.enable_full_saves == true ) {
                    sprintf(filename, "%s/output%08u", PhysiCell_settings.folder.c_str(),  PhysiCell_globals.full_output_index);
                    save_PhysiCell_to_MultiCellDS_v2(filename, microenvironment, PhysiCell_globals.current_time);
                }

                // save legacy simulation report
                if (PhysiCell_settings.enable_legacy_saves == true) {
                    log_output(PhysiCell_globals.current_time, PhysiCell_globals.full_output_index, microenvironment, report_file);
                }
            }

            // save svg plot if it's time
            if ((PhysiCell_settings.enable_SVG_saves == true) and (svg_countdown < 0.5 * diffusion_dt)) {
                svg_countdown += PhysiCell_settings.SVG_save_interval;
                PhysiCell_globals.SVG_output_index++;

                // save ssvg cross section
                std::vector<std::string> (*cell_coloring_function)(Cell*) = my_coloring_function;  // bue 20240130: not going global
                std::string (*substrate_coloring_function)(double, double, double) = paint_by_density_percentage;  // bue 20241026: not going global
                sprintf(filename, "%s/snapshot%08u.svg", PhysiCell_settings.folder.c_str(), PhysiCell_globals.SVG_output_index);
                SVG_plot(filename, microenvironment, 0.0, PhysiCell_globals.current_time, cell_coloring_function, substrate_coloring_function);
            }
        }
    // ===== RESTORE OUTPUT =====
    dup2(stdout_fd, STDOUT_FILENO);
    dup2(stderr_fd, STDERR_FILENO);
    close(stdout_fd);
    close(stderr_fd);
    // ==========================

    //} catch (const std::exception& e) {  // reference to the base of a polymorphic object
    //    std::cout << e.what();  // information from length_error printed
    //}

    // go home
    return PyLong_FromLong(0);
}


// extended Python C++ function stop
static PyObject* physicell_stop(PyObject *self, PyObject *args) {

    // save final data simulation snapshot
    sprintf(filename, "%s/final", PhysiCell_settings.folder.c_str());
    save_PhysiCell_to_MultiCellDS_v2(filename, microenvironment, PhysiCell_globals.current_time);

    // save final svg cross section
    std::vector<std::string> (*cell_coloring_function)(Cell*) = my_coloring_function;  // bue 20240130: not going global
    std::string (*substrate_coloring_function)(double, double, double) = paint_by_density_percentage;  // bue 20241026: not going global
    sprintf(filename, "%s/final.svg", PhysiCell_settings.folder.c_str());
    SVG_plot(filename, microenvironment, 0.0, PhysiCell_globals.current_time, cell_coloring_function, substrate_coloring_function);


    // save legacy simulation report
    if (PhysiCell_settings.enable_legacy_saves == true) {
        log_output(PhysiCell_globals.current_time, PhysiCell_globals.full_output_index, microenvironment, report_file);
        report_file.close();
    }

    // go home
    return PyLong_FromLong(0);
}


// extend Python C++ function set_parameter
static PyObject* physicell_set_parameter(PyObject *self, PyObject *args) {
    // extract input
    const char *label;
    PyObject *value;
    if (! PyArg_ParseTuple(args, "sO", &label, &value)) {
        return NULL;
    }

    // recall from C++ into Python variable
    int parindex;
    char error[1024];

    // boole
    parindex = parameters.bools.find_index(label);
    if (parindex > -1) {
        Parameter<bool> *bool_parameter = &parameters.bools[parindex];
        bool_parameter->value = PyLong_AsLong(value);

    } else {
        // int
        parindex = parameters.ints.find_index(label);
        if (parindex > -1) {
            Parameter<int> *int_parameter = &parameters.ints[parindex];
            int_parameter->value = PyLong_AsLong(value);

        } else {
            // float
            parindex = parameters.doubles.find_index(label);
            if (parindex > -1) {
                Parameter<double> *float_parameter = &parameters.doubles[parindex];
                float_parameter->value = PyFloat_AsDouble(value);

            } else {
                // str
                parindex = parameters.strings.find_index(label);
                if (parindex > -1) {
                    if (! PyUnicode_Check(value)) {
                        snprintf(error, sizeof(error), "Error: %s value cannot be interpreted as a string!", label);
                        PyErr_SetString(PyExc_ValueError, error);
                        return NULL;
                    }
                    Parameter<std::string> *str_parameter = &parameters.strings[parindex];
                    str_parameter->value = PyUnicode_AsUTF8(value);

                } else {
                    //error
                    snprintf(error, sizeof(error), "Error: unknown parameter! %s", label);
                    PyErr_SetString(PyExc_KeyError, error);
                    return NULL;
                }
            }
        }
    }
    // go home
    return PyLong_FromLong(0);
}


// extend Python C++ function get_parameter
static PyObject* physicell_get_parameter(PyObject *self, PyObject *args) {
    // extract variable label
    const char *label;
    if (! PyArg_ParseTuple(args, "s", &label)) {
        return NULL;
    }

    // recall from C++ into Python variable
    int parindex;

    // bool
    parindex = parameters.bools.find_index(label);
    if (parindex > -1) {
        Parameter<bool> bool_parameter = parameters.bools[parindex];
        return PyBool_FromLong(bool_parameter.value);

    } else {
        // int
        parindex = parameters.ints.find_index(label);
        if (parindex > -1) {
            Parameter<int> int_parameter = parameters.ints[parindex];
            return PyLong_FromLong(int_parameter.value);

        } else {
            // float
            parindex = parameters.doubles.find_index(label);
            if (parindex > -1) {
                Parameter<double> float_parameter = parameters.doubles[parindex];
                return PyFloat_FromDouble(float_parameter.value);

            } else {
                // str
                parindex = parameters.strings.find_index(label);
                if (parindex > -1) {
                    Parameter<std::string> str_parameter = parameters.strings[parindex];
                    return PyUnicode_FromString(str_parameter.value.c_str());

                } else {
                    //error
                    char error[1024];
                    snprintf(error, sizeof(error), "Error: unknown parameter! %s", label);
                    PyErr_SetString(PyExc_KeyError, error);
                    return NULL;
                }
            }
        }
    }
}


// extended Python C++ function set_variable
static PyObject* physicell_set_variable(PyObject *self, PyObject *args) {
    // extract variable label and value
    const char *label;
    double value;
    if (! PyArg_ParseTuple(args, "sd", &label, &value)) {
        return NULL;
    }

    // store
    for (Cell* pCell : (*all_cells)) {
        //set_single_behavior(pCell, "custom:<label>" , value); // bue20240206: trouble child.
        int varindex = pCell->custom_data.find_variable_index(label);
        if (varindex > -1) {
            pCell->custom_data[label] = value;
        } else {
            char error[1024];
            snprintf(error, sizeof(error), "Error: unknown custom_data variable! %s", label);
            PyErr_SetString(PyExc_KeyError, error);
            return NULL;
        }
    }

    // going home
    return PyLong_FromLong(0);
}


// extended Python C++ function get_variable
static PyObject* physicell_get_variable(PyObject *self, PyObject *args) {
    // extract variable label
    const char *label;
    if (! PyArg_ParseTuple(args, "s", &label)) {
        return NULL;
    }

    // recall from C++ intp Python list
    int cell_count = all_cells->size();
    PyObject *pList = PyList_New(cell_count);
    for (int i=0; i < cell_count; i++) {
        Cell* pCell = (*all_cells)[i];
        int varindex = pCell->custom_data.find_variable_index(label);
        if (varindex > -1) {
            double value = pCell->custom_data[label];
            PyList_SetItem(pList, i, PyFloat_FromDouble(value));
        } else {
            Py_XDECREF(pList);
            char error[1024];
            snprintf(error, sizeof(error), "Error: unknown custom_data variable! %s", label);
            PyErr_SetString(PyExc_KeyError, error);
            return NULL;
        }
    }

    // going home
    return pList;
}


// extended Python C++ function set_vector
static PyObject* physicell_set_vector(PyObject *self, PyObject *args) {
    // https://stackoverflow.com/questions/22458298/extending-python-with-c-pass-a-list-to-pyarg-parsetuple
    // https://docs.python.org/3/c-api/arg.html

    // extract variable label and vector
    const char *label;
    PyObject *pList;
    std::vector<double> value;
    if (! PyArg_ParseTuple(args, "sO!", &label, &PyList_Type, &pList)) {
        return NULL;
    }

    // transfrom Python list of numbers to C++ vector of doubles
    PyObject *pItem;
    for (int i=0; i < PyList_Size(pList); i++) {
        pItem = PyList_GetItem(pList, i);
        if (!(PyNumber_Check(pItem))) {
            PyErr_SetString(PyExc_TypeError, "Error: all list items must be integer or float!");
            return NULL;
        } else {
            value.push_back(PyFloat_AsDouble(pItem));
        }
    }

    // store
    for (Cell* pCell : (*all_cells)) {
        int vectindex = pCell->custom_data.find_vector_variable_index(label);
        if (vectindex > -1) {
            pCell->custom_data.vector_variables[vectindex].value = value;
        } else {
            char error[1024];
            snprintf(error, sizeof(error), "Error: unknown custom_data vector! %s", label);
            PyErr_SetString(PyExc_KeyError, error);
            return NULL;
        }
    }

    // going home
    return PyLong_FromLong(0);
}


// extended Python C++ function get_vector
static PyObject* physicell_get_vector(PyObject *self, PyObject *args) {
    // extract variable label
    const char *label;
    if (! PyArg_ParseTuple(args, "s", &label)) {
        return NULL;
    }

    // recall from C++ into Python list of list
    bool first = true;
    int cell_count = all_cells->size();
    PyObject *pLlist = PyList_New(cell_count);
    for (int i=0; i < cell_count; i++) {
        Cell* pCell = (*all_cells)[i];
        int vectindex = pCell->custom_data.find_vector_variable_index(label);
        if (first) {
            first = false;
            if (vectindex < 0) {
                Py_XDECREF(pLlist);
                char error[1024];
                snprintf(error, sizeof(error), "Error: unknown custom_data vector! %s", label);
                PyErr_SetString(PyExc_KeyError, error);
                return NULL;
            }
        }
        std::vector<double> vector = pCell->custom_data.vector_variables[vectindex].value;
        int element_count = vector.size();
        PyObject *pList = PyList_New(element_count);
        for (int j=0; j < element_count; j++) {
            double value = vector[j];
            PyList_SetItem(pList, j, PyFloat_FromDouble(value));
        }
        PyList_SetItem(pLlist, i, pList);
    }

    // going home
    return pLlist;
}


// extended Python C++ function get_cell
static PyObject* physicell_get_cell(PyObject *self, PyObject *args) {

    // recall from C++ into Python list of list
    int cell_count = all_cells->size();
    PyObject *pLlist = PyList_New(cell_count);

    for (int i=0; i < cell_count; i++) {
        Cell* pCell = (*all_cells)[i];
        PyObject *pList = PyList_New(6);  // id, x, y, z, dead, cell_type
        PyList_SetItem(pList, 0, PyLong_FromLong(pCell->ID)); // id
        PyList_SetItem(pList, 1, PyFloat_FromDouble(pCell->position[0])); // x
        PyList_SetItem(pList, 2, PyFloat_FromDouble(pCell->position[1])); // y
        PyList_SetItem(pList, 3, PyFloat_FromDouble(pCell->position[2])); // z
        PyList_SetItem(pList, 4, PyFloat_FromDouble(pCell->phenotype.death.dead)); // dead
        PyList_SetItem(pList, 5, PyUnicode_FromString((pCell->type_name).c_str())); // cell_type
        PyList_SetItem(pLlist, i, pList);
    }

    // going home
    return pLlist;
}


// extended Python C++ function get_microenv
static PyObject* physicell_get_microenv(PyObject *self, PyObject *args) {
    // extract variable label
    const char *substrate;
    if (! PyArg_ParseTuple(args, "s", &substrate)) {
        return NULL;
    }

    // extract substrate index
    int subsindex = microenvironment.find_density_index(substrate);
    if (subsindex < 0) {
        char error[1024];
        snprintf(error, sizeof(error), "Error: unknown substrate! %s", substrate);
        PyErr_SetString(PyExc_KeyError, error);
        return NULL;
    }

    // recall from C++ into Python list of list
    int voxel_count = microenvironment.number_of_voxels();
    PyObject *pLlist = PyList_New(voxel_count);

    for (int n=0; n < voxel_count; n++) {
        PyObject *pList = PyList_New(4);  // x, y, z, conc
        PyList_SetItem(pList, 0, PyFloat_FromDouble(microenvironment.mesh.voxels[n].center[0]));  // x
        PyList_SetItem(pList, 1, PyFloat_FromDouble(microenvironment.mesh.voxels[n].center[1]));  // y
        PyList_SetItem(pList, 2, PyFloat_FromDouble(microenvironment.mesh.voxels[n].center[2]));  // z
        PyList_SetItem(pList, 3, PyFloat_FromDouble(microenvironment(n)[subsindex]));  // conc
        PyList_SetItem(pLlist, n, pList);
    }

    // going home
    return pLlist;
}


// extended Python C++ function system
static PyObject* physicell_system(PyObject *self, PyObject *args) {
    // variables
    const char *command;
    int sts;

    // extract and run commandline command
    if (! PyArg_ParseTuple(args, "s",  &command)) {
        return NULL;
    }
    sts = system(command);

    // going home
    return PyLong_FromLong(sts);
}


// method table lists method name and address
static struct PyMethodDef ExtendpyMethods[] = {
    {"start", physicell_start, METH_VARARGS,
     "input:\n    settingxml 'path/to/setting.xml' file (string); default is 'config/PhysiCell_settings.xml'.\n    reload (bool) density and parameter structs; default is False.\n\noutput:\n    PhysiCell processing. 0 for success.\n\nrun:\n    from extending import physicell\n    physicell.start('path/to/setting.xml')\n\ndescription:\n    function (re)initializes PhysiCell as specified in the settings.xml, cells.csv, and cell_rules.csv files and generates the step zero observation output."
    },
    {"step", physicell_step, METH_VARARGS,
     "input:\n    none.\n\noutput:\n    PhysiCell processing. 0 for success.\n\nrun:\n    from extending import physicell\n    physicell.step()\n\ndescription:\n    function runs one time step."
    },
    {"stop", physicell_stop, METH_VARARGS,
     "input:\n    none.\n\noutput:\n    PhysiCell processing. 0 for success.\n\nrun:\n    from extending import physicell\n    physicell.stop()\n\ndescription:\n    function finalizes a PhysiCell episode."
    },
    {"set_parameter", physicell_set_parameter, METH_VARARGS,
     "input:\n    parameter name (string), vector value (bool or int or float or str).\n\noutput:\n    0 for success and -1 for failure.\n\nrun:\n    from extending import physicell\n    physicell.set_parameter('my_parameter', value)\n\ndescription:\n    function to store a user parameter."
    },
    {"get_parameter", physicell_get_parameter, METH_VARARGS,
     "input:\n    parameter name (string)\n\noutput:\n    values (bool or int or float or str).\n\nrun:\n    from extending import physicell\n    physicell.get_parameter('my_parameter')\n\ndescription:\n    function to recall a user parameter."
    },
    {"set_variable", physicell_set_variable, METH_VARARGS,
     "input:\n    variable name (string), variable value (float or integer).\n\noutput:\n    0 for success and -1 for failure.\n\nrun:\n    from extending import physicell\n    physicell.set_variable('my_variable', value)\n\ndescription:\n    function to store a custom variable value."
    },
    {"get_variable", physicell_get_variable, METH_VARARGS,
     "input:\n    variable name (string).\n\noutput:\n    values (list of floats).\n\nrun:\n    from extending import physicell\n    physicell.get_variable('my_variable')\n\ndescription:\n    function to recall a custom variable."
    },
    {"set_vector", physicell_set_vector, METH_VARARGS,
     "input:\n    vector name (string), vector values (list of floats or integers).\n\noutput:\n    0 for success and -1 for failure.\n\nrun:\n    from extending import physicell\n    physicell.set_vector('my_vector', value)\n\ndescription:\n    function to store a custom vector."
    },
    {"get_vector", physicell_get_vector, METH_VARARGS,
     "input:\n    vector name (string)\n\noutput:\n    values (list of list of floats).\n\nrun:\n    from extending import physicell\n    physicell.get_vector('my_vector')\n\ndescription:\n    function to recall a custom vector."
    },
    {"get_cell", physicell_get_cell, METH_VARARGS,
     "input:\n    none\n\noutput:\n    values (list of list of floats).\n\nrun:\n    from extending import physicell\n    physicell.get_cell()\n\ndescription:\n    function to recall cell position coordinate and id."
    },
    {"get_microenv", physicell_get_microenv, METH_VARARGS,
     "input:\n    substrate name (string)\n\noutput:\n    values (list of list of floats).\n\nrun:\n    from extending import physicell\n    physicell.get_microenv('my_substrate')\n\ndescription:\n    function to recall a voxel center coordinates and substrate concentration."
    },
    {"system", physicell_system, METH_VARARGS, "execute a shell command."},
    /*{NULL, NULL, 0, NULL}  // Sentinel */
};


// module definition structure
static struct PyModuleDef physicellmodule = {
    PyModuleDef_HEAD_INIT,
    "physicell",  // name of the module
    NULL,  //physicell_doc,  // module documentation, may be NULL
    -1,  // size of per-interpreter state of the module, or -1 if he module keeps state in global variables.
    ExtendpyMethods
};


// pass module structure to the Python interpreter initializatin function
PyMODINIT_FUNC PyInit_physicell(void) {
    return PyModule_Create(&physicellmodule);
}


// off we go
int main(int argc, char *argv[]) {
    // get program name
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0].\n");
        exit(1);
    }

    // add the builtin module to the initailzation table
    if (PyImport_AppendInittab("physicell", PyInit_physicell) == -1) {
        fprintf(stderr, "Error: could not extend in-built modules table.\n");
        exit(1);
    }

    // pass argv[0] to the Python interpreter
    Py_SetProgramName(program);

    // initialize the Python interpreter. required. if this step fails, it will be a fatal error
    Py_Initialize();

    // optional import the module.
    // alternatively, import can be deferred until the embedded script imports it.
    PyObject *pmodule = PyImport_ImportModule("physicell");
    if (!pmodule) {
        PyErr_Print();
        fprintf(stderr, "Error: could not import module 'physicell'.\n");
    }

    // free up memory
    PyMem_RawFree(program);
    return 0;
}

