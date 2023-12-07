#include "UiPanelViewOutput.h"

#include "Config.h"
#include "Camera.h"
#include "ModelSplatsDevice.h"
#include "Trainer.cuh"
#include "UiFrame.h"

using namespace std;

Project& UiPanelViewOutput::getProject() const {
    return *frame.project;
}

UiPanelViewOutput::UiPanelViewOutput(wxWindow *parent, UiFrame& frame, wxGLContext* context) : wxPanel(parent),
    frame(frame), context(context) {
    sizer = new wxBoxSizer(wxVERTICAL);

    canvas = new wxGLCanvas(this);
    canvas->SetCurrent(*context);
    sizer->Add(canvas, wxSizerFlags().Shaped().Expand());

    renderer = new FboRenderer(RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y);

    wxBoxSizer* sizerLower = new wxBoxSizer(wxHORIZONTAL);
    sizer->Add(sizerLower, wxSizerFlags().Expand());

    textIterations = new wxStaticText(this, wxID_ANY, "");
    sizerLower->Add(textIterations, wxSizerFlags(1));
    textSplats = new wxStaticText(this, wxID_ANY, "");
    sizerLower->Add(textSplats, wxSizerFlags(1));

    sizerLower->Fit(this);

    SetSizerAndFit(sizer);
}

UiPanelViewOutput::~UiPanelViewOutput() {
    delete renderer;
}

void UiPanelViewOutput::refreshProject() {
    refreshText();
}

void UiPanelViewOutput::refreshText() {
    textSplats->SetLabel(std::to_string(frame.trainer->model->count) + " / " +
        std::to_string(frame.trainer->model->capacity) + " total splats");
    textIterations->SetLabel(std::to_string(getProject().iterations) + " training iterations");
}

void UiPanelViewOutput::render() {
    canvas->SetCurrent(*context);

    frame.trainer->render(renderer->frameBuffer, RENDER_RESOLUTION_X, RENDER_RESOLUTION_Y,
                               getProject().previewSplatScale, Camera::getPreviewCamera(getProject()));

    renderer->render(canvas->GetSize().x, canvas->GetSize().y);
    canvas->SwapBuffers();
}

void UiPanelViewOutput::onPaint(wxPaintEvent& event) {
    if (!IsShown()) return;
    render();
}

void UiPanelViewOutput::onIdle(wxIdleEvent& event) {
    render();
    event.RequestMore();
}

BEGIN_EVENT_TABLE(UiPanelViewOutput, wxPanel)
EVT_PAINT(UiPanelViewOutput::onPaint)
EVT_IDLE(UiPanelViewOutput::onIdle)
END_EVENT_TABLE()
