#pragma once

#include <wx/wx.h>
#include <wx/spinctrl.h>

#include "Project.h"
#include "ui/UiFrame.h"

class UiPanelToolsTruth : public wxPanel {

private:
    UiFrame& frame;
    Project& getProject() const;

    wxStaticBoxSizer* sizerSphere1;
    wxStaticBoxSizer* sizerSphere2;
    wxBoxSizer* sizerControls;

    wxSpinCtrl* spinSphere1Count;
    wxSpinCtrlDouble* spinSphere1Distance;
    wxSpinCtrlDouble* spinSphere1Fov;
    wxSpinCtrlDouble* spinSphere1RotX;
    wxSpinCtrlDouble* spinSphere1RotY;

    wxSpinCtrl* spinSphere2Count;
    wxSpinCtrlDouble* spinSphere2Distance;
    wxSpinCtrlDouble* spinSphere2Fov;
    wxSpinCtrlDouble* spinSphere2RotX;
    wxSpinCtrlDouble* spinSphere2RotY;

    wxSpinCtrl* spinRtSamples;
    wxButton* buttonRandomRotate;
    wxButton* buttonCapture;

    enum CameraSphereIds {
        S1_COUNT,
        S1_DISTANCE,
        S1_FOV,
        S1_ROTX,
        S1_ROTY,
        S2_COUNT,
        S2_DISTANCE,
        S2_FOV,
        S2_ROTX,
        S2_ROTY
    };

public:
    UiPanelToolsTruth(wxWindow* parent, UiFrame& frame);

    void refreshProject();

    void onSpin(wxSpinEvent& event);
    void onSpinDouble(wxSpinDoubleEvent& event);

    void onSpinRtSamples(wxSpinEvent& event);
    void onButtonCapture(wxCommandEvent& event);
    void onButtonRandomRotate(wxCommandEvent& event);

};
