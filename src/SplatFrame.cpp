#include "SplatFrame.h"

SplatFrame::SplatFrame() :
        wxFrame(nullptr, wxID_ANY, "Gaussian Splatterer", wxDefaultPosition, {1280, 720}, wxDEFAULT_FRAME_STYLE) {
    panel = new wxPanel(this);
    sizer = new wxBoxSizer(wxHORIZONTAL);
    panel->SetSizerAndFit(sizer);

    truthCameras = new TruthCameras();

    panelInput = new SplatPanelInput(panel);
    sizer->Add(panelInput, wxSizerFlags(1).Shaped().Expand().Border());

    panelTools = new SplatPanelTools(panel);
    sizer->Add(panelTools, wxSizerFlags(0).Border());
}

SplatFrame::~SplatFrame() {
    delete truthCameras;
}
