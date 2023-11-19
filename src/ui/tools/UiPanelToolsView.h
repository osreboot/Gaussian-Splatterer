#pragma once

#include <wx/wx.h>
#include <wx/spinctrl.h>

#include "Project.h"
#include "ui/UiFrame.h"

class UiPanelToolsView : public wxPanel {

private:
    UiFrame& getFrame() const;
    Project& getProject() const;

    wxStaticBoxSizer* sizer;

    wxCheckBox* checkPreviewCamera;
    wxSpinCtrl* spinPreviewCamera;
    wxSpinCtrl* spinPreviewRtSamples;

    wxSpinCtrl* spinRenderResX;
    wxSpinCtrl* spinRenderResY;

    wxButton* buttonRenderRtx;
    wxButton* buttonRenderSplats;

public:
    UiPanelToolsView(wxWindow* parent);

    void refreshProject();
    void refreshCameraCount();

    void onCheckPreviewCamera(wxCommandEvent& event);
    void onSpinPreviewCamera(wxSpinEvent& event);
    void onSpinPreviewRtSamples(wxSpinEvent& event);

    void onSpinRenderRes(wxSpinEvent& event);

    void onButtonRenderRtx(wxCommandEvent& event);
    void onButtonRenderSplats(wxCommandEvent& event);

};
