#include "UiFrame.h"

using namespace std;

UiFrame::UiFrame() :
        wxFrame(nullptr, wxID_ANY, "Gaussian Splatterer", wxDefaultPosition, {1280, 720}, wxDEFAULT_FRAME_STYLE) {
    panel = new wxPanel(this);
    sizer = new wxBoxSizer(wxHORIZONTAL);
    panel->SetSizerAndFit(sizer);

    timeLastUpdate = chrono::high_resolution_clock::now();

    truthCameras = new TruthCameras();

    rtx = new RtxHost({RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y});
    rtx->load(R"(C:\Users\Calvin\Desktop\Archives\Development\Resources\Gecko 3d model\Gecko_m1.obj)",
              R"(C:\Users\Calvin\Desktop\Archives\Development\Resources\Gecko 3d model\Files\Textures\Gecko Body Texture.BMP)");
    trainer = new Trainer();

    panelInput = new UiPanelInput(panel);
    sizer->Add(panelInput, wxSizerFlags(1).Shaped().Expand().Border());

    panelOutput = new UiPanelOutput(panel, panelInput->context);
    sizer->Add(panelOutput, wxSizerFlags(1).Shaped().Expand().Border());

    panelTools = new UiPanelTools(panel);
    sizer->Add(panelTools, wxSizerFlags(0).Border());
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
        autoTrainingBudget = min(1.0f, autoTrainingBudget + delta * 10.0f);

        if(autoTrainingBudget >= 1.0f) {
            autoTrainingBudget = 0.0f;
            if((trainer->iterations + 1) % 50 == 0) {
                wxCommandEvent eventFake = wxCommandEvent(wxEVT_NULL, 0);
                panelTools->onButtonCamerasRotRandom(eventFake);
                panelTools->onButtonCamerasCapture(eventFake);
            }
            trainer->train((trainer->iterations + 1) % 100 == 0);
            panelTools->updateIterationCount();
            panelTools->updateSplatCount();
        }
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
