#include "UiFrame.h"
#include "UiDialogAbout.h"

using namespace std;

UiFrame::UiFrame() :
        wxFrame(nullptr, wxID_ANY, "Gaussian Splatterer", wxDefaultPosition, {1280, 720}, wxDEFAULT_FRAME_STYLE) {
    panel = new wxPanel(this);
    sizer = new wxBoxSizer(wxHORIZONTAL);
    panel->SetSizerAndFit(sizer);

    timeLastUpdate = chrono::high_resolution_clock::now();

    truthCameras = new TruthCameras();

    rtx = new RtxHost({RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y});
    rtx->load(R"(C:\Users\Calvin\Desktop\Archives\Development\Resources\Gecko 3d model\Splats\Gecko.obj)",
              R"(C:\Users\Calvin\Desktop\Archives\Development\Resources\Gecko 3d model\Splats\Gecko.BMP)");
    trainer = new Trainer();

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
    menuFileSave->Append(FILE_SAVE_SPLATS, "Save Splats");
    menuFileSave->Append(FILE_SAVE_SETTINGS, "Save Settings");
    menuFileLoad->Append(FILE_LOAD_SPLATS, "Load Splats");
    menuFileLoad->Append(FILE_LOAD_SETTINGS, "Load Settings");

    menuAbout = new wxMenu();
    menuBar->Append(menuAbout, "About");
    menuAbout->Append(ABOUT_ABOUT, "About");

    menuBar->Bind(wxEVT_COMMAND_TOOL_CLICKED, &UiFrame::onMenuButton, this);

    panelInput = new UiPanelInput(panel);
    panelInput->SetMinSize({256, 256});
    sizer->Add(panelInput, wxSizerFlags(1).Shaped().Expand().Border());

    panelOutput = new UiPanelOutput(panel, panelInput->context);
    panelOutput->SetMinSize({256, 256});
    sizer->Add(panelOutput, wxSizerFlags(1).Shaped().Expand().Border());

    panelTools = new UiPanelTools(panel);
    sizer->Add(panelTools, wxSizerFlags(0).Border());

    sizer->Fit(this);
}

UiFrame::~UiFrame() {
    delete truthCameras;
    delete rtx;
}

void UiFrame::update() {
    timeNow = chrono::high_resolution_clock::now();
    float delta = (float)chrono::duration_cast<chrono::nanoseconds>(timeNow - timeLastUpdate).count() / 1000000000.0f;
    //delta = min(delta, 0.2f);
    timeLastUpdate = timeNow;

    truthCameras->update(delta);

    if(autoTraining) {
        autoTrainingBudget = min(1.0f, autoTrainingBudget + delta * 100.0f);

        if(autoTrainingBudget >= 1.0f) {
            autoTrainingBudget = 0.0f;
            if((trainer->iterations + 1) % 50 == 0) {
                wxCommandEvent eventFake = wxCommandEvent(wxEVT_NULL, 0);
                panelTools->onButtonCamerasRotRandom(eventFake);
                panelTools->onButtonCamerasCapture(eventFake);
            }
            trainer->train((trainer->iterations + 1) % 200 == 0);
            panelTools->updateIterationCount();
            panelTools->updateSplatCount();
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
