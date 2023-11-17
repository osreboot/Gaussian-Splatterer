#pragma once

#include <wx/wx.h>

#include "ui/UiFrame.h"
#include "Project.h"

class UiPanelToolsTrain : public wxPanel{

private:
    UiFrame& getFrame() const;
    Project& getProject() const;

    wxStaticBoxSizer* sizer;

    wxBoxSizer* sizerAuto;
    wxBoxSizer* sizerManual;

    wxButton* buttonAutoStart;
    wxButton* buttonAutoStop;

public:
    UiPanelToolsTrain(wxWindow* parent);

    void onButtonAutoStart(wxCommandEvent& event);
    void onButtonAutoStop(wxCommandEvent& event);

};
