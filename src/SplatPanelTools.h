#pragma once

#include <wx/wx.h>
#include <wx/spinctrl.h>

class SplatPanelTools : public wxPanel {

private:
    wxBoxSizer* sizer;

    wxStaticBoxSizer* sizerStaticInput;
    wxStaticBoxSizer* sizerStaticTruth;
    wxStaticBoxSizer* sizerStaticTrain;
    wxStaticBoxSizer* sizerStaticOutput;

    wxSpinCtrl* spinCtrlCamerasCount;
    wxSpinCtrlDouble* spinCtrlCamerasDistance;

public:
    SplatPanelTools(wxWindow *parent);

    void onSpinCtrlCamerasCount(wxSpinEvent& event);
    void onSpinCtrlCamerasDistance(wxSpinDoubleEvent& event);

};
