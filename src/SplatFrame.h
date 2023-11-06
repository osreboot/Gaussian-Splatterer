#pragma once

#include <wx/wx.h>

#include "SplatPanelInput.h"
#include "SplatPanelTools.h"

class SplatFrame : public wxFrame {

private:
    wxPanel* panel;
    wxBoxSizer* sizer;

public:
    SplatPanelInput* panelInput;
    SplatPanelTools* panelTools;

    SplatFrame();

private:

};