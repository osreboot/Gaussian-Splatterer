#include "UiFrame.h"

#include <fstream>

#include "UiPanelViewInput.h"
#include "UiPanelViewOutput.h"
#include "UiPanelTools.h"
#include "UiDialogAbout.h"

#include "Config.h"
#include "Project.h"
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

    rtx = new RtxHost({RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y});
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
            if((project->iterations + 1) % 50 == 0) {
                wxCommandEvent eventFake = wxCommandEvent(wxEVT_NULL, 0);
                panelTools->panelTruth->onButtonRandomRotate(eventFake);
                panelTools->panelTruth->onButtonCapture(eventFake);
            }
            trainer->train(*project, (project->iterations + 1) % 200 == 0);
            panelOutput->refreshText();
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

}

void UiFrame::loadSettings(const std::string& path) {
    nlohmann::json j;
    std::ifstream file(path);
    file >> j;
    nlohmann::from_json(j, *project);
    refreshProject();
}

void UiFrame::loadSplats(const std::string& path) {

}

void UiFrame::onMenuButton(wxCommandEvent& event) {
    if (event.GetId() == FILE_SAVE_SPLATS) {
        wxLogMessage("TODO");
    } else if (event.GetId() == FILE_SAVE_SETTINGS) {

        wxFileDialog dialog(this, "Save Settings", "", "settings", "JSON Files (*.json)|*.json", wxFD_SAVE | wxFD_OVERWRITE_PROMPT);
        if (dialog.ShowModal() == wxID_CANCEL) return;
        saveSettings(dialog.GetPath().ToStdString());

    } else if (event.GetId() == FILE_LOAD_SPLATS) {
        wxLogMessage("TODO");
    } else if (event.GetId() == FILE_LOAD_SETTINGS) {

        wxFileDialog dialog(this, "Load Settings", "", "settings", "JSON Files (*.json)|*.json", wxFD_OPEN | wxFD_FILE_MUST_EXIST);
        if (dialog.ShowModal() == wxID_CANCEL) return;
        loadSettings(dialog.GetPath().ToStdString());

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
