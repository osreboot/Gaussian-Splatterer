#include "UiPanelToolsView.h"

#include <cuda_runtime.h>
#include <stb/stb_image_write.h>

#include "rtx/RtxHost.h"
#include "ui/UiFrame.h"
#include "ui/UiPanelTools.h"
#include "Camera.h"
#include "Project.h"
#include "Trainer.cuh"

UiFrame& UiPanelToolsView::getFrame() const {
    return *dynamic_cast<UiFrame*>(GetParent()->GetParent()->GetParent());
}

Project& UiPanelToolsView::getProject() const {
    return *getFrame().project;
}

UiPanelToolsView::UiPanelToolsView(wxWindow* parent) : wxPanel(parent) {
    sizer = new wxStaticBoxSizer(wxHORIZONTAL, this, "4. Visualize Splats");



    wxBoxSizer* sizerControls = new wxBoxSizer(wxVERTICAL);
    sizer->Add(sizerControls, wxSizerFlags().Border());

    sizerControls->Add(new wxStaticText(this, wxID_ANY, "View RT Samples"), wxSizerFlags().Border(wxUP));
    spinCamRtSamples = new wxSpinCtrl(this);
    spinCamRtSamples->SetRange(1, 5000);
    spinCamRtSamples->SetMinSize({64, -1});
    spinCamRtSamples->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &UiPanelToolsView::onSpinCamRtSamples, this);
    sizerControls->Add(spinCamRtSamples);



    wxStaticBoxSizer* sizerCamRef = new wxStaticBoxSizer(wxVERTICAL, this, "View Truth");
    sizer->Add(sizerCamRef, wxSizerFlags().Border());

    checkCamRef = new wxCheckBox(this, wxID_ANY, "Enabled");
    checkCamRef->Bind(wxEVT_COMMAND_CHECKBOX_CLICKED, &UiPanelToolsView::onCheckCamRef, this);
    sizerCamRef->Add(checkCamRef, wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));

    sizerCamRef->Add(new wxStaticText(this, wxID_ANY, "Truth Camera"), wxSizerFlags().Border(wxUP | wxLEFT | wxRIGHT));
    spinCamRefIdx = new wxSpinCtrl(this);
    spinCamRefIdx->SetMinSize({64, -1});
    spinCamRefIdx->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &UiPanelToolsView::onSpinCamRefIdx, this);
    spinCamRefIdx->Disable();
    sizerCamRef->Add(spinCamRefIdx, wxSizerFlags().Border(wxDOWN | wxLEFT | wxRIGHT));



    wxStaticBoxSizer* sizerCamFree = new wxStaticBoxSizer(wxHORIZONTAL, this, "View Free");
    sizer->Add(sizerCamFree, wxSizerFlags().Border());

    wxBoxSizer* sizerCamFree1 = new wxBoxSizer(wxVERTICAL);
    sizerCamFree->Add(sizerCamFree1, wxSizerFlags().Border());
    wxBoxSizer* sizerCamFree2 = new wxBoxSizer(wxVERTICAL);
    sizerCamFree->Add(sizerCamFree2, wxSizerFlags().Border());
    wxBoxSizer* sizerCamFree3 = new wxBoxSizer(wxVERTICAL);
    sizerCamFree->Add(sizerCamFree3, wxSizerFlags().Border());

    sizerCamFree1->Add(new wxStaticText(this, wxID_ANY, "Distance"));
    spinCamFreeDistance = new wxSpinCtrlDouble(this, F_DISTANCE);
    spinCamFreeDistance->SetRange(0.1, 40.0);
    spinCamFreeDistance->SetDigits(2);
    spinCamFreeDistance->SetIncrement(0.1);
    spinCamFreeDistance->SetMinSize({64, -1});
    sizerCamFree1->Add(spinCamFreeDistance);

    sizerCamFree1->Add(new wxStaticText(this, wxID_ANY, "FOV (Y Deg.)"), wxSizerFlags().Border(wxUP));
    spinCamFreeFov = new wxSpinCtrlDouble(this, F_FOV);
    spinCamFreeFov->SetRange(10.0, 120.0);
    spinCamFreeFov->SetDigits(1);
    spinCamFreeFov->SetIncrement(0.1);
    spinCamFreeFov->SetMinSize({64, -1});
    sizerCamFree1->Add(spinCamFreeFov);

    sizerCamFree2->Add(new wxStaticText(this, wxID_ANY, "Rotation (X,Y)"));
    spinCamFreeRotX = new wxSpinCtrlDouble(this, F_ROTX);
    spinCamFreeRotX->SetRange(0.0, 360.0);
    spinCamFreeRotX->SetDigits(1);
    spinCamFreeRotX->SetIncrement(5);
    spinCamFreeRotX->SetMinSize({64, -1});
    sizerCamFree2->Add(spinCamFreeRotX);
    spinCamFreeRotY = new wxSpinCtrlDouble(this, F_ROTY);
    spinCamFreeRotY->SetRange(0.0, 360.0);
    spinCamFreeRotY->SetDigits(1);
    spinCamFreeRotY->SetIncrement(5);
    spinCamFreeRotY->SetMinSize({64, -1});
    sizerCamFree2->Add(spinCamFreeRotY);

    checkCamFreeOrbit = new wxCheckBox(this, wxID_ANY, "Orbit");
    checkCamFreeOrbit->Bind(wxEVT_COMMAND_CHECKBOX_CLICKED, &UiPanelToolsView::onCheckCamFreeOrbit, this);
    sizerCamFree3->Add(checkCamFreeOrbit);

    sizerCamFree3->Add(new wxStaticText(this, wxID_ANY, "Orbit Speed"), wxSizerFlags().Border(wxUP));
    spinCamFreeOrbitSpeed = new wxSpinCtrlDouble(this, F_SPEED);
    spinCamFreeOrbitSpeed->SetRange(0.0, 10.0);
    spinCamFreeOrbitSpeed->SetDigits(4);
    spinCamFreeOrbitSpeed->SetIncrement(0.1);
    spinCamFreeOrbitSpeed->SetMinSize({64, -1});
    sizerCamFree3->Add(spinCamFreeOrbitSpeed);



    wxStaticBoxSizer* sizerRender = new wxStaticBoxSizer(wxHORIZONTAL, this, "Render Static Image");
    sizer->Add(sizerRender, wxSizerFlags().Border());

    wxBoxSizer* sizerRender1 = new wxBoxSizer(wxVERTICAL);
    sizerRender->Add(sizerRender1, wxSizerFlags().Border());

    sizerRender1->Add(new wxStaticText(this, wxID_ANY, "Resolution (X/Y)"));
    spinRenderResX = new wxSpinCtrl(this);
    spinRenderResX->SetRange(16, 8192);
    spinRenderResX->SetMinSize({64, -1});
    spinRenderResX->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &UiPanelToolsView::onSpinRenderRes, this);
    sizerRender1->Add(spinRenderResX);
    spinRenderResY = new wxSpinCtrl(this);
    spinRenderResY->SetRange(16, 8192);
    spinRenderResY->SetMinSize({64, -1});
    spinRenderResY->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &UiPanelToolsView::onSpinRenderRes, this);
    sizerRender1->Add(spinRenderResY);

    wxBoxSizer* sizerRender2 = new wxBoxSizer(wxVERTICAL);
    sizerRender->Add(sizerRender2, wxSizerFlags().Border());

    buttonRenderRtx = new wxButton(this, wxID_ANY, "Render Ray Tracer");
    buttonRenderRtx->SetMinSize({-1, 40});
    buttonRenderRtx->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelToolsView::onButtonRenderRtx, this);
    sizerRender2->Add(buttonRenderRtx, wxSizerFlags().Expand().Border(wxUP | wxLEFT | wxRIGHT));

    buttonRenderSplats = new wxButton(this, wxID_ANY, "Render Splats");
    buttonRenderSplats->SetMinSize({-1, 40});
    buttonRenderSplats->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiPanelToolsView::onButtonRenderSplats, this);
    sizerRender2->Add(buttonRenderSplats, wxSizerFlags().Expand().Border(wxDOWN | wxLEFT | wxRIGHT));



    Bind(wxEVT_COMMAND_SPINCTRLDOUBLE_UPDATED, &UiPanelToolsView::onSpinDouble, this);



    SetSizerAndFit(sizer);
}

void UiPanelToolsView::refreshProject() {
    spinCamRtSamples->SetValue(getProject().previewRtSamples);

    checkCamRef->SetValue(getProject().previewTruth);
    spinCamRefIdx->SetValue(getProject().previewTruthIndex);

    checkCamFreeOrbit->SetValue(getProject().previewFreeOrbit);
    spinCamFreeOrbitSpeed->SetValue(getProject().previewFreeOrbitSpeed);
    spinCamFreeDistance->SetValue(getProject().previewFreeDistance);
    spinCamFreeFov->SetValue(getProject().previewFreeFovDeg);
    spinCamFreeRotX->SetValue(getProject().previewFreeRotX);
    spinCamFreeRotY->SetValue(getProject().previewFreeRotY);

    spinRenderResX->SetValue(getProject().renderResX);
    spinRenderResY->SetValue(getProject().renderResY);

    refreshCameraCount();
    refreshViewPanels();
}

void UiPanelToolsView::refreshCameraCount() {
    spinCamRefIdx->SetRange(1, Camera::getCamerasCount(getProject()));
    getProject().previewTruthIndex = std::max(0, std::min(Camera::getCamerasCount(getProject()) - 1, getProject().previewTruthIndex));
}

void UiPanelToolsView::refreshViewPanels() {
    spinCamRefIdx->Enable(getProject().previewTruth);
    checkCamFreeOrbit->Enable(!getProject().previewTruth);
    spinCamFreeOrbitSpeed->Enable(!getProject().previewTruth && getProject().previewFreeOrbit);
    spinCamFreeDistance->Enable(!getProject().previewTruth);
    spinCamFreeFov->Enable(!getProject().previewTruth);
    spinCamFreeRotX->Enable(!getProject().previewTruth);
    spinCamFreeRotY->Enable(!getProject().previewTruth);
}

void UiPanelToolsView::onCheckCamRef(wxCommandEvent& event) {
    getProject().previewTruth = event.IsChecked();
    refreshViewPanels();
}

void UiPanelToolsView::onSpinCamRefIdx(wxSpinEvent& event) {
    getProject().previewTruthIndex = event.GetValue() - 1;
}

void UiPanelToolsView::onSpinCamRtSamples(wxSpinEvent& event) {
    getProject().previewRtSamples = event.GetValue();
}

void UiPanelToolsView::onCheckCamFreeOrbit(wxCommandEvent& event) {
    getProject().previewFreeOrbit = event.IsChecked();
    getProject().previewTimer = 0.0f;
    refreshViewPanels();
}

void UiPanelToolsView::onSpinDouble(wxSpinDoubleEvent& event) {
    switch(event.GetId()) {
        case SpinDoubleIds::F_SPEED: getProject().previewFreeOrbitSpeed = (float)event.GetValue(); break;
        case SpinDoubleIds::F_DISTANCE: getProject().previewFreeDistance = (float)event.GetValue(); break;
        case SpinDoubleIds::F_FOV: getProject().previewFreeFovDeg = (float)event.GetValue(); break;
        case SpinDoubleIds::F_ROTX: getProject().previewFreeRotX = (float)event.GetValue(); break;
        case SpinDoubleIds::F_ROTY: getProject().previewFreeRotY = (float)event.GetValue(); break;
        default: break;
    }
}

void UiPanelToolsView::onSpinRenderRes(wxSpinEvent& event) {
    getProject().renderResX = spinRenderResX->GetValue();
    getProject().renderResY = spinRenderResY->GetValue();
}

void UiPanelToolsView::onButtonRenderRtx(wxCommandEvent& event) {
    wxFileDialog dialog(this, "Save Image", "", "RayTracer", "PNG Files (*.png)|*.png", wxFD_SAVE | wxFD_OVERWRITE_PROMPT);
    if (dialog.ShowModal() == wxID_CANCEL) return;

    uint32_t* frameBuffer;
    cudaMallocManaged(&frameBuffer, getProject().renderResX * getProject().renderResY * sizeof(uint32_t));

    getFrame().rtx->render(frameBuffer, {getProject().renderResX, getProject().renderResY},
                           Camera::getPreviewCamera(getProject()), {0.0f, 0.0f, 0.0f}, getProject().rtSamples, {});

    stbi_flip_vertically_on_write(true);
    stbi_write_png(dialog.GetPath().ToStdString().c_str(), getProject().renderResX, getProject().renderResY, 4,
                   frameBuffer, getProject().renderResX * sizeof(uint32_t));

    cudaFree(frameBuffer);
}

void UiPanelToolsView::onButtonRenderSplats(wxCommandEvent& event) {
    wxFileDialog dialog(this, "Save Image", "", "GaussianSplats", "PNG Files (*.png)|*.png", wxFD_SAVE | wxFD_OVERWRITE_PROMPT);
    if (dialog.ShowModal() == wxID_CANCEL) return;

    uint32_t* frameBuffer;
    cudaMallocManaged(&frameBuffer, getProject().renderResX * getProject().renderResY * sizeof(uint32_t));

    getFrame().trainer->render(frameBuffer, getProject().renderResX, getProject().renderResY, Camera::getPreviewCamera(getProject()));

    stbi_flip_vertically_on_write(true);
    stbi_write_png(dialog.GetPath().ToStdString().c_str(), getProject().renderResX, getProject().renderResY, 4,
                   frameBuffer, getProject().renderResX * sizeof(uint32_t));

    cudaFree(frameBuffer);
}
