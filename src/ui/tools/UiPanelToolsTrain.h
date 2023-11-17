#pragma once

#include <wx/wx.h>

class UiPanelToolsTrain : public wxPanel{

private:
    wxStaticBoxSizer* sizer;

    wxBoxSizer* sizerAuto;
    wxBoxSizer* sizerManual;

    wxButton* buttonAutoStart;
    wxButton* buttonAutoStop;

public:
    UiPanelToolsTrain(wxWindow* parent);

};
