#pragma once

#include <wx/wx.h>
#include <wx/spinctrl.h>

class UiPanelToolsTruth : public wxPanel {

private:
    wxStaticBoxSizer* sizer;

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

    wxButton* buttonRandomRotate;
    wxButton* buttonCapture;
    wxStaticText* textStatus;

public:
    UiPanelToolsTruth(wxWindow* parent);

    void onSpinSphere1Count(wxSpinEvent& event);
    void onSpinSphere1Distance(wxSpinDoubleEvent& event);
    void onSpinSphere1Fov(wxSpinDoubleEvent& event);
    void onSpinSphere1RotX(wxSpinDoubleEvent& event);
    void onSpinSphere1RotY(wxSpinDoubleEvent& event);

    void onSpinSphere2Count(wxSpinEvent& event);
    void onSpinSphere2Distance(wxSpinDoubleEvent& event);
    void onSpinSphere2Fov(wxSpinDoubleEvent& event);
    void onSpinSphere2RotX(wxSpinDoubleEvent& event);
    void onSpinSphere2RotY(wxSpinDoubleEvent& event);

    void onButtonRandomRotate(wxCommandEvent& event);
    void onButtonCapture(wxCommandEvent& event);

};
