#include "SplatFrame.h"

class SplatApp : public wxApp {

public:
    bool OnInit() override {
        SplatFrame* frame = new SplatFrame();
        frame->Show();
        return true;
    }

};

wxIMPLEMENT_APP(SplatApp);
