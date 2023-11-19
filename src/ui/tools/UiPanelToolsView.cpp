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

    checkPreviewCamera = new wxCheckBox(this, wxID_ANY, "View Truth Camera");
    checkPreviewCamera->Bind(wxEVT_COMMAND_CHECKBOX_CLICKED, &UiPanelToolsView::onCheckPreviewCamera, this);
    sizerControls->Add(checkPreviewCamera);

    sizerControls->Add(new wxStaticText(this, wxID_ANY, "View Camera"), wxSizerFlags().Border(wxUP));
    spinPreviewCamera = new wxSpinCtrl(this);
    spinPreviewCamera->SetMinSize({64, -1});
    spinPreviewCamera->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &UiPanelToolsView::onSpinPreviewCamera, this);
    spinPreviewCamera->Disable();
    sizerControls->Add(spinPreviewCamera, wxSizerFlags().Border(wxDOWN));

    sizerControls->Add(new wxStaticText(this, wxID_ANY, "View RT Samples"), wxSizerFlags().Border(wxUP));
    spinPreviewRtSamples = new wxSpinCtrl(this);
    spinPreviewRtSamples->SetRange(1, 5000);
    spinPreviewRtSamples->SetMinSize({64, -1});
    spinPreviewRtSamples->Bind(wxEVT_COMMAND_SPINCTRL_UPDATED, &UiPanelToolsView::onSpinPreviewRtSamples, this);
    sizerControls->Add(spinPreviewRtSamples);



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



    SetSizerAndFit(sizer);
}

void UiPanelToolsView::refreshProject() {
    checkPreviewCamera->SetValue(getProject().previewIndex != -1);
    spinPreviewCamera->Enable(getProject().previewIndex != -1);
    spinPreviewCamera->SetValue(getProject().previewIndex + 1);
    spinPreviewRtSamples->SetValue(getProject().previewRtSamples);
    refreshCameraCount();
    spinRenderResX->SetValue(getProject().renderResX);
    spinRenderResY->SetValue(getProject().renderResY);
}

void UiPanelToolsView::refreshCameraCount() {
    spinPreviewCamera->SetRange(1, Camera::getCamerasCount(getProject()));
    getProject().previewIndex = std::max(-1, std::min(Camera::getCamerasCount(getProject()) - 1, getProject().previewIndex));
}

void UiPanelToolsView::onCheckPreviewCamera(wxCommandEvent& event) {
    if(event.IsChecked()) {
        getProject().previewIndex = spinPreviewCamera->GetValue() - 1;
        spinPreviewCamera->Enable();
    } else {
        getProject().previewIndex = -1;
        spinPreviewCamera->Disable();
    }
}

void UiPanelToolsView::onSpinPreviewCamera(wxSpinEvent& event) {
    getProject().previewIndex = event.GetValue() - 1;
}

void UiPanelToolsView::onSpinPreviewRtSamples(wxSpinEvent& event) {
    getProject().previewRtSamples = event.GetValue();
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
