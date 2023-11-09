#include "ui/UiFrame.h"

class SplatApp : public wxApp {

public:
    bool OnInit() override {
        UiFrame* frame = new UiFrame();
        frame->Show();
        return true;
    }

};

wxIMPLEMENT_APP(SplatApp);
