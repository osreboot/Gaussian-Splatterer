#include "UiFrame.h"

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

    rtx = new RtxHost({RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y});
    rtx->load(R"(C:\Users\Calvin\Desktop\Archives\Development\Resources\Gecko 3d model\Splats\Gecko.obj)",
              R"(C:\Users\Calvin\Desktop\Archives\Development\Resources\Gecko 3d model\Splats\Gecko.BMP)");

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
    panelInput->SetMinSize({256, 256});
    sizerViews->Add(panelInput, wxSizerFlags().Shaped().Expand().Border());

    panelOutput = new UiPanelViewOutput(panel, panelInput->context);
    panelOutput->SetMinSize({256, 256});
    sizerViews->Add(panelOutput, wxSizerFlags().Shaped().Expand().Border());

    panelTools = new UiPanelTools(panel);
    sizer->Add(panelTools, wxSizerFlags().Border());

    sizer->Fit(this);
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
            if((trainer->iterations + 1) % 50 == 0) {
                wxCommandEvent eventFake = wxCommandEvent(wxEVT_NULL, 0);
                //panelTools->onButtonCamerasRotRandom(eventFake);
                //panelTools->onButtonCamerasCapture(eventFake);
            }
            trainer->train((trainer->iterations + 1) % 200 == 0);
            //panelTools->updateIterationCount();
            //panelTools->updateSplatCount();
        }
    }
}

void UiFrame::onMenuButton(wxCommandEvent& event) {
    if (event.GetId() == FILE_SAVE_SPLATS) {
        wxLogMessage("TODO");
    } else if (event.GetId() == FILE_SAVE_SETTINGS) {
        wxLogMessage("TODO");
    } else if (event.GetId() == FILE_LOAD_SPLATS) {
        wxLogMessage("TODO");
    } else if (event.GetId() == FILE_LOAD_SETTINGS) {
        wxLogMessage("TODO");
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
