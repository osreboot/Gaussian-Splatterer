#include "UiPanelViewInput.h"

#include "Config.h"
#include "Camera.h"
#include "Project.h"
#include "rtx/RtxHost.h"
#include "Trainer.cuh"

Project& UiPanelViewInput::getProject() const {
    return *frame.project;
}

UiPanelViewInput::UiPanelViewInput(wxWindow *parent, UiFrame& frame) : wxPanel(parent), frame(frame) {
    sizer = new wxBoxSizer(wxVERTICAL);

    canvas = new wxGLCanvas(this);
    context = new wxGLContext(canvas);
    canvas->SetCurrent(*context);
    sizer->Add(canvas, wxSizerFlags().Shaped().Expand());

    renderer = new FboRenderer(RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y);

    textFrames = new wxStaticText(this, wxID_ANY, "No captured frames");
    sizer->Add(textFrames, wxSizerFlags().Expand());

    SetSizerAndFit(sizer);
}

UiPanelViewInput::~UiPanelViewInput() {
    delete context;
    delete renderer;
}

void UiPanelViewInput::refreshProject() {
    refreshText();
}

void UiPanelViewInput::refreshText() {
    textFrames->SetLabel(std::to_string(frame.trainer->truthFrameBuffersW.size()) + " saved truth frames (x2, both black and white background variants of each)");
}

void UiPanelViewInput::render() {
    canvas->SetCurrent(*context);

    frame.rtx->render(renderer->frameBuffer, {RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y},
                           Camera::getPreviewCamera(getProject()), {0.0f, 0.0f, 0.0f}, getProject().previewRtSamples,
                           getProject().previewTruth ? std::vector<Camera>() : Camera::getCameras(getProject()));

    renderer->render(canvas->GetSize().x, canvas->GetSize().y);
    canvas->SwapBuffers();
}

void UiPanelViewInput::onPaint(wxPaintEvent& event) {
    if (!IsShown()) return;
    render();
}

void UiPanelViewInput::onIdle(wxIdleEvent& event) {
    render();
    event.RequestMore();
}

BEGIN_EVENT_TABLE(UiPanelViewInput, wxPanel)
EVT_PAINT(UiPanelViewInput::onPaint)
EVT_IDLE(UiPanelViewInput::onIdle)
END_EVENT_TABLE()


