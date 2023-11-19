#include "UiPanelViewInput.h"

#include "Config.h"
#include "Camera.h"
#include "Project.h"
#include "rtx/RtxHost.h"
#include "Trainer.cuh"

UiFrame& UiPanelViewInput::getFrame() const {
    return *dynamic_cast<UiFrame*>(GetParent()->GetParent());;
}

Project& UiPanelViewInput::getProject() const {
    return *getFrame().project;
}

UiPanelViewInput::UiPanelViewInput(wxWindow *parent) : wxPanel(parent) {
    sizer = new wxBoxSizer(wxVERTICAL);

    sizer->Add(new wxStaticText(this, wxID_ANY, "RTX Ray Tracer View"));

    canvas = new wxGLCanvas(this);
    canvas->SetMinSize({256, 256});
    context = new wxGLContext(canvas);
    canvas->SetCurrent(*context);
    sizer->Add(canvas, wxSizerFlags().Shaped().Expand());

    renderer = new FboRenderer(RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y);

    textFrames = new wxStaticText(this, wxID_ANY, "No captured frames");
    sizer->Add(textFrames);

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
    textFrames->SetLabel(std::to_string(getFrame().trainer->truthFrameBuffersW.size()) + " saved truth frames (x2, both black and white background variants of each)");
}

void UiPanelViewInput::render() {
    canvas->SetCurrent(*context);

    getFrame().rtx->render(renderer->frameBuffer, {RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y},
                           Camera::getPreviewCamera(getProject()), {0.0f, 0.0f, 0.0f},
                           getProject().previewIndex == -1 ? Camera::getCameras(getProject()) : std::vector<Camera>());

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


