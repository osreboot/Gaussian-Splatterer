#include "UiPanelViewOutput.h"

#include "Config.h"
#include "Camera.h"
#include "ModelSplatsDevice.h"
#include "Trainer.cuh"
#include "UiFrame.h"

using namespace std;

UiFrame& UiPanelViewOutput::getFrame() const {
    return *dynamic_cast<UiFrame*>(GetParent()->GetParent());;
}

Project& UiPanelViewOutput::getProject() const {
    return *getFrame().project;
}

UiPanelViewOutput::UiPanelViewOutput(wxWindow *parent, wxGLContext* context) : wxPanel(parent), context(context) {
    sizer = new wxBoxSizer(wxVERTICAL);

    sizer->Add(new wxStaticText(this, wxID_ANY, "Gaussian Splats View"));

    canvas = new wxGLCanvas(this);
    canvas->SetMinSize({256, 256});
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
    textSplats->SetLabel(std::to_string(getFrame().trainer->model->count) + " / " +
        std::to_string(getFrame().trainer->model->capacity) + " Gaussian splats");
    textIterations->SetLabel(std::to_string(getProject().iterations) + " training iterations");
}

void UiPanelViewOutput::render() {
    canvas->SetCurrent(*context);

    getFrame().trainer->render(renderer->frameBuffer, Camera::getPreviewCamera(getProject()));

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
