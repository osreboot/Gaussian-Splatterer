#include "UiFrame.h"

#include <fstream>
#include <wx/progdlg.h>

#include "UiPanelViewInput.h"
#include "UiPanelViewOutput.h"
#include "UiPanelTools.h"
#include "dialog/UiDialogAbout.h"

#include "Config.h"
#include "Project.h"
#include "ModelSplatsDevice.h"
#include "ModelSplatsHost.h"
#include "rtx/RtxHost.h"
#include "Trainer.cuh"

using namespace std;

UiFrame::UiFrame() :
        wxFrame(nullptr, wxID_ANY, "Gaussian Splatterer") {
    panel = new wxPanel(this);
    sizer = new wxBoxSizer(wxVERTICAL);
    panel->SetSizerAndFit(sizer);

    project = new Project();
    project->sphere2.fovDeg = 30.0f;
    project->pathModel = R"(C:\Users\Calvin\Desktop\Archives\Development\Resources\Gecko 3d model\Splats\Gecko.obj)";
    project->pathTexture = R"(C:\Users\Calvin\Desktop\Archives\Development\Resources\Gecko 3d model\Splats\Gecko.BMP)";

    rtx = new RtxHost();
    rtx->load(*project);

    trainer = new Trainer();

    timeLastUpdate = chrono::high_resolution_clock::now();

    menuBar = new wxMenuBar();
    SetMenuBar(menuBar);

    menuFile = new wxMenu();
    menuFileInit = new wxMenu();
    menuFileLoad = new wxMenu();
    menuFileSave = new wxMenu();

    menuBar->Append(menuFile, "File");
    menuFile->AppendSubMenu(menuFileInit, "New");
    menuFile->AppendSeparator();
    menuFile->AppendSubMenu(menuFileSave, "Save");
    menuFile->AppendSubMenu(menuFileLoad, "Load");
    menuFileSave->Append(FILE_SAVE_PROJECT, "Save Project");
    menuFileSave->AppendSeparator();
    menuFileSave->Append(FILE_SAVE_SPLATS, "Save Splats");
    menuFileSave->Append(FILE_SAVE_SETTINGS, "Save Settings");
    menuFileLoad->Append(FILE_LOAD_PROJECT, "Load Project");
    menuFileLoad->AppendSeparator();
    menuFileLoad->Append(FILE_LOAD_SPLATS, "Load Splats");
    menuFileLoad->Append(FILE_LOAD_SETTINGS, "Load Settings");

    menuAbout = new wxMenu();
    menuBar->Append(menuAbout, "About");
    menuAbout->Append(ABOUT_ABOUT, "About");

    menuBar->Bind(wxEVT_COMMAND_TOOL_CLICKED, &UiFrame::onMenuButton, this);

    sizerViews = new wxBoxSizer(wxHORIZONTAL);
    sizer->Add(sizerViews, wxSizerFlags(1).Expand());

    panelInput = new UiPanelViewInput(panel);
    sizerViews->Add(panelInput, wxSizerFlags(1).Expand().Border());

    panelOutput = new UiPanelViewOutput(panel, panelInput->context);
    sizerViews->Add(panelOutput, wxSizerFlags(1).Expand().Border());

    panelTools = new UiPanelTools(panel);
    sizer->Add(panelTools, wxSizerFlags().Border());

    sizer->Fit(this);

    refreshProject();
}

UiFrame::~UiFrame() {
    delete project;

    delete rtx;
    delete trainer;
}

void UiFrame::update() {
    timeNow = chrono::high_resolution_clock::now();
    float delta = (float)chrono::duration_cast<chrono::nanoseconds>(timeNow - timeLastUpdate).count() / 1000000000.0f;
    //delta = min(delta, 0.2f);
    timeLastUpdate = timeNow;

    project->previewTimer += delta;

    if(autoTraining) {
        autoTrainingBudget = min(1.0f, autoTrainingBudget + delta * 100.0f);

        if(autoTrainingBudget >= 1.0f) {
            autoTrainingBudget = 0.0f;

            const bool capture = project->intervalCapture > 0 && project->iterations % project->intervalCapture == 0;
            const bool densify = project->intervalDensify > 0 && project->iterations % project->intervalDensify == 0;

            if(capture) {
                wxCommandEvent eventFake = wxCommandEvent(wxEVT_NULL, 0);
                panelTools->panelTruth->onButtonRandomRotate(eventFake);
                panelTools->panelTruth->onButtonCapture(eventFake);
            }
            trainer->train(*project, densify);
            panelOutput->refreshText();
            panelTools->panelTrain->refreshText();
        }
    }
}

void UiFrame::refreshProject() {
    // TODO load new model here
    // TODO nuke trainer captures

    // TODO stop auto training
    // TODO stop auto training in nested classes

    panelInput->refreshProject();
    panelOutput->refreshProject();
    panelTools->refreshProject();
}

void UiFrame::saveSettings(const std::string& path) const {
    nlohmann::json j;
    nlohmann::to_json(j, *project);

    std::ofstream file(path);
    file << j;
}

void UiFrame::saveSplats(const std::string& path) const {
    wxProgressDialog dialog("Saving Gaussian Splats", "Writing splats to \"" + path + "\"...", trainer->model->count + 1000, panel, wxPD_AUTO_HIDE);
    ModelSplatsHost model(*trainer->model);

    int progress = 1000;
    dialog.Update(progress);

    std::ofstream file(path);
    for (int i = 0; i < model.count; i++) {
        file << "v " << model.locations[i * 3] << " " << model.locations[i * 3 + 1] << " " << model.locations[i * 3 + 2] << "\n";

        file << "sh";
        for (int f = 0; f < model.shCoeffs * 3; f++) {
            file << " " << model.shs[i * 3 * model.shCoeffs + f];
        }
        file << "\n";

        file << "s " << model.scales[i * 3] << " " << model.scales[i * 3 + 1] << " " << model.scales[i * 3 + 2] << "\n";
        file << "a " << model.opacities[i] << "\n";
        file << "r " << model.rotations[i * 4] << " " << model.rotations[i * 4 + 1] << " " << model.rotations[i * 4 + 2] << " " << model.rotations[i * 4 + 3] << "\n";

        dialog.Update(++progress);
    }
}

void UiFrame::loadSettings(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        wxMessageDialog dialog(this, "Failed to load settings file at \"" + path + "\"!", "Load Failed!", wxICON_ERROR);
        dialog.ShowModal();
        return;
    }

    nlohmann::json j;
    std::ifstream file(path);
    file >> j;
    nlohmann::from_json(j, *project);
}

void UiFrame::loadSplats(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        wxMessageDialog dialog(this, "Failed to load splats file at \"" + path + "\"!", "Load Failed!", wxICON_ERROR);
        dialog.ShowModal();
        return;
    }

    // Count the number of lines in the file so we can get a progress estimate
    std::ifstream fileLines(path);
    int linesCount = (int)std::count(std::istreambuf_iterator<char>(fileLines), std::istreambuf_iterator<char>(), '\n');

    wxProgressDialog dialog("Loading Gaussian Splats", "Loading splats from \"" + path + "\"...", linesCount + 2000, panel, wxPD_AUTO_HIDE);

    int progress = 0;

    std::optional<int> shCoeffs = nullopt;

    std::vector<float> locations;
    std::vector<float> shs;
    std::vector<float> scales;
    std::vector<float> opacities;
    std::vector<float> rotations;

    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            for (int f = 0; f < 3; f++) {
                float x;
                iss >> x;
                locations.push_back(x);
            }
        } else if (prefix == "sh") {
            int shCoeffsCount = 0;
            float x;
            while (iss >> x) {
                shs.push_back(x);
                shCoeffsCount++;
            }
            if (!shCoeffs) shCoeffs = shCoeffsCount;
            else assert(shCoeffs == shCoeffsCount);
        } else if (prefix == "s") {
            for (int f = 0; f < 3; f++) {
                float x;
                iss >> x;
                scales.push_back(x);
            }
        } else if (prefix == "a") {
            float x;
            iss >> x;
            opacities.push_back(x);
        } else if (prefix == "r") {
            for (int f = 0; f < 4; f++) {
                float x;
                iss >> x;
                rotations.push_back(x);
            }
        }

        dialog.Update(++progress);
    }

    ModelSplatsHost modelHost(locations, shs, scales, opacities, rotations);

    progress += 1000;
    dialog.Update(progress);

    delete trainer->model;
    trainer->model = new ModelSplatsDevice(modelHost);
}

void UiFrame::onMenuButton(wxCommandEvent& event) {
    if (event.GetId() == FILE_SAVE_PROJECT) {

        wxDirDialog dialog(this, "Save Project", "");
        if (dialog.ShowModal() == wxID_CANCEL) return;
        saveSettings(dialog.GetPath().ToStdString() + "\\settings.json");
        saveSplats(dialog.GetPath().ToStdString() + "\\splats.gobj");

    }else if (event.GetId() == FILE_SAVE_SPLATS) {

        wxFileDialog dialog(this, "Save Gaussian Splats", "", "splats", "Gaussian OBJ Files (*.gobj)|*.gobj", wxFD_SAVE | wxFD_OVERWRITE_PROMPT);
        if (dialog.ShowModal() == wxID_CANCEL) return;
        saveSplats(dialog.GetPath().ToStdString());

    } else if (event.GetId() == FILE_SAVE_SETTINGS) {

        wxFileDialog dialog(this, "Save Settings", "", "settings", "JSON Files (*.json)|*.json", wxFD_SAVE | wxFD_OVERWRITE_PROMPT);
        if (dialog.ShowModal() == wxID_CANCEL) return;
        saveSettings(dialog.GetPath().ToStdString());

    } else if (event.GetId() == FILE_LOAD_PROJECT) {

        wxDirDialog dialog(this, "Load Project", "", wxDD_DIR_MUST_EXIST);
        if (dialog.ShowModal() == wxID_CANCEL) return;
        loadSettings(dialog.GetPath().ToStdString() + "\\settings.json");
        loadSplats(dialog.GetPath().ToStdString() + "\\splats.gobj");
        refreshProject();

    } else if (event.GetId() == FILE_LOAD_SPLATS) {

        wxFileDialog dialog(this, "Load Gaussian Splats", "", "splats", "Gaussian OBJ Files (*.gobj)|*.gobj", wxFD_OPEN | wxFD_FILE_MUST_EXIST);
        if (dialog.ShowModal() == wxID_CANCEL) return;
        loadSplats(dialog.GetPath().ToStdString());
        refreshProject();

    } else if (event.GetId() == FILE_LOAD_SETTINGS) {

        wxFileDialog dialog(this, "Load Settings", "", "settings", "JSON Files (*.json)|*.json", wxFD_OPEN | wxFD_FILE_MUST_EXIST);
        if (dialog.ShowModal() == wxID_CANCEL) return;
        loadSettings(dialog.GetPath().ToStdString());
        refreshProject();

    } else if (event.GetId() == ABOUT_ABOUT) {
        UiDialogAbout dialog(this);
    }
}

void UiFrame::onPaint(wxPaintEvent& event) {
    if(!IsShown()) return;
    update();
}

void UiFrame::onIdle(wxIdleEvent& event) {
    update();
    event.RequestMore();
}

BEGIN_EVENT_TABLE(UiFrame, wxFrame)
EVT_PAINT(UiFrame::onPaint)
EVT_IDLE(UiFrame::onIdle)
END_EVENT_TABLE()
