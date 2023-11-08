#pragma once

#include <wx/wx.h>

#include "TruthCameras.h"
#include "SplatPanelInput.h"
#include "SplatPanelTools.h"

class SplatFrame : public wxFrame {

private:
    wxPanel* panel;
    wxBoxSizer* sizer;

public:
    TruthCameras* truthCameras;

    SplatPanelInput* panelInput;
    SplatPanelTools* panelTools;

    SplatFrame();
    ~SplatFrame() override;

private:

};