#pragma once

#include "UiFrame.h"

class UiDialogAbout : public wxDialog {

public:
    UiDialogAbout(wxWindow* parent);

private:
    void onButtonClose(wxCommandEvent& event);

};
