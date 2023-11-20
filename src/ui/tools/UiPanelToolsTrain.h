#pragma once

#include <wx/wx.h>
#include <wx/spinctrl.h>

#include "ui/UiFrame.h"
#include "Project.h"

class UiPanelToolsTrain : public wxPanel{

private:
    UiFrame& getFrame() const;
    Project& getProject() const;

    wxStaticBoxSizer* sizer;

    wxButton* buttonAutoStart;
    wxButton* buttonAutoStop;

    wxButton* buttonManual1;
    wxButton* buttonManual5;
    wxButton* buttonManual10;
    wxButton* buttonManual20;
    wxButton* buttonManual50;
    wxButton* buttonManual100;
    wxButton* buttonManual200;
    wxButton* buttonManualDensify;

    wxStaticText* textIntervalCapture;
    wxStaticText* textIntervalDensify;
    wxSpinCtrl* spinIntervalCapture;
    wxSpinCtrl* spinIntervalDensify;

    enum ManualIds {
        M_1,
        M_5,
        M_10,
        M_20,
        M_50,
        M_100,
        M_200,
        M_D,
        M_END
    };

public:
    UiPanelToolsTrain(wxWindow* parent);

    void refreshProject();
    void refreshText();

    void onButtonAutoStart(wxCommandEvent& event);
    void onButtonAutoStop(wxCommandEvent& event);

    void onButtonManual(wxCommandEvent& event);

    void onSpinIntervalCapture(wxSpinEvent& event);
    void onSpinIntervalDensify(wxSpinEvent& event);

};
