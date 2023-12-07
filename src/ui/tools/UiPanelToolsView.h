#pragma once

#include <wx/wx.h>
#include <wx/spinctrl.h>

#include "Project.h"
#include "ui/UiFrame.h"

class UiPanelToolsView : public wxPanel {

private:
    UiFrame& frame;
    Project& getProject() const;

    wxSpinCtrl* spinCamRtSamples;
    wxSpinCtrlDouble* spinSplatScale;

    wxCheckBox* checkCamRef;
    wxSpinCtrl* spinCamRefIdx;

    wxCheckBox* checkCamFreeOrbit;
    wxSpinCtrlDouble* spinCamFreeOrbitSpeed;
    wxSpinCtrlDouble* spinCamFreeDistance;
    wxSpinCtrlDouble* spinCamFreeFov;
    wxSpinCtrlDouble* spinCamFreeRotX;
    wxSpinCtrlDouble* spinCamFreeRotY;

    wxSpinCtrl* spinRenderResX;
    wxSpinCtrl* spinRenderResY;

    wxButton* buttonRenderRtx;
    wxButton* buttonRenderSplats;

    enum SpinDoubleIds {
        F_SPEED,
        F_DISTANCE,
        F_FOV,
        F_ROTX,
        F_ROTY
    };

public:
    UiPanelToolsView(wxWindow* parent, UiFrame& frame);

    void refreshProject();
    void refreshCameraCount();
    void refreshViewPanels();

    void onCheckCamRef(wxCommandEvent& event);
    void onSpinCamRefIdx(wxSpinEvent& event);
    void onSpinCamRtSamples(wxSpinEvent& event);
    void onSpinSplatScale(wxSpinDoubleEvent& event);

    void onCheckCamFreeOrbit(wxCommandEvent& event);

    void onSpinDouble(wxSpinDoubleEvent& event);

    void onSpinRenderRes(wxSpinEvent& event);

    void onButtonRenderRtx(wxCommandEvent& event);
    void onButtonRenderSplats(wxCommandEvent& event);

};
