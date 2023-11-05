#include "SplatApp.h"
#include "SplatFrame.h"

bool SplatApp::OnInit() {
    SplatFrame* frame = new SplatFrame("Guassian Splatterer");
    frame->Show();

    return true;
}
