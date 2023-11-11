#pragma once

#include <wx/wx.h>

#include "rtx/RtxHost.h"
#include "Trainer.cuh"
#include "TruthCameras.h"
#include "UiPanelInput.h"
#include "UiPanelOutput.h"
#include "UiPanelTools.h"

class UiFrame : public wxFrame {

private:
    std::chrono::high_resolution_clock::time_point timeLastUpdate, timeNow;

    wxPanel* panel;
    wxBoxSizer* sizer;

    float autoTrainingBudget = 0.0f;

public:
    TruthCameras* truthCameras;

    RtxHost* rtx;
    Trainer* trainer;

    UiPanelInput* panelInput;
    UiPanelOutput* panelOutput;
    UiPanelTools* panelTools;

    bool autoTraining = false;

    UiFrame();
    ~UiFrame() override;

private:
    void update();

    void onPaint(wxPaintEvent& event);
    void onIdle(wxIdleEvent& event);

    DECLARE_EVENT_TABLE();

};