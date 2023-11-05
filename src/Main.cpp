#include <chrono>
#include <glfw/glfw3.h>

#include <wx/wxprec.h>

#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif

#include "Window.h"
#include "rtx/RtxHost.h"
#include "wx/SplatApp.h"

using namespace std;

chrono::high_resolution_clock::time_point timeLastUpdate, timeNow;

int main(int argc, char** argv){
    // Create the display and initialize the ray tracing program
    Window window(1920, 1080, "Gaussian Splatterer");
    RtxHost rtx;
    rtx.setSplatModel(R"(C:\Users\Calvin\Desktop\Archives\Development\Resources\Gecko 3d model\Gecko_m1.obj)",
                      R"(C:\Users\Calvin\Desktop\Archives\Development\Resources\Gecko 3d model\Files\Textures\Gecko Body Texture.BMP)");

    wxApp* splatApp = new SplatApp();
    wxApp::SetInstance(splatApp);
    wxEntryStart(argc, argv);
    wxTheApp->OnInit();

    timeLastUpdate = chrono::high_resolution_clock::now();

    // Loop while the user hasn't clicked the window close button
    while(!window.exiting()){
        // Delta is the time in seconds since the last update. This value is used to synchronize scene timing elements
        // that need to run at a consistent speed, even through inconsistent rendering speeds / update timings (lag).

        timeNow = chrono::high_resolution_clock::now();
        float delta = (float)chrono::duration_cast<chrono::nanoseconds>(timeNow - timeLastUpdate).count() / 1000000000.0f;
        //delta = min(delta, 0.2f);
        timeLastUpdate = timeNow;

        // Prepare the display for rendering
        window.preUpdate();

        // Run the ray tracing program
        rtx.update(delta, window.getSize().x, window.getSize().y, (uint64_t)window.getFrameBuffer());

        // Render the final image to the display
        window.postUpdate();

        wxTheApp->OnRun();
    }

    glfwTerminate();

    wxTheApp->OnExit();
    wxEntryCleanup();

    return 0;
}