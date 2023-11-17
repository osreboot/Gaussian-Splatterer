#pragma once

#include <wx/wx.h>

class UiDialogAbout : public wxDialog {

public:
    explicit UiDialogAbout(wxWindow* parent);

private:
    void onButtonClose(wxCommandEvent& event);

};
