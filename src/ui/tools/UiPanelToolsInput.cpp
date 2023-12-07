#include "UiPanelToolsInput.h"

#include <wx/progdlg.h>
#include <fstream>

#include "ui/UiFrame.h"
#include "ui/UiPanelViewInput.h"
#include "Camera.h"
#include "Project.h"
#include "Trainer.cuh"
#include "rtx/RtxHost.h"

Project& UiPanelToolsInput::getProject() const {
    return *frame.project;
}

UiPanelToolsInput::UiPanelToolsInput(wxWindow* parent, UiFrame& frame) : wxPanel(parent), frame(frame) {
    wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);

    sizer->Add(new wxStaticText(this, wxID_ANY, "Model"), wxSizerFlags().Expand().Border(wxUP | wxLEFT | wxRIGHT));
    fileModel = new wxFilePickerCtrl(this, wxID_ANY, "", "Load Model", "OBJ Files (*.obj)|*.obj",
                                     wxDefaultPosition, wxDefaultSize, wxFLP_USE_TEXTCTRL | wxFLP_OPEN | wxFLP_SMALL);
    fileModel->Bind(wxEVT_FILEPICKER_CHANGED, &UiPanelToolsInput::onFileModel, this);
    sizer->Add(fileModel, wxSizerFlags().Expand().Border(wxDOWN | wxLEFT | wxRIGHT));

    sizer->Add(new wxStaticText(this, wxID_ANY, "Diffuse Texture"), wxSizerFlags().Expand().Border(wxUP | wxLEFT | wxRIGHT));
    fileTextureDiffuse = new wxFilePickerCtrl(this, wxID_ANY, "", "Load Diffuse Texture",
                                              "PNG Files (*.png)|*.png|JPEG Files (*.jpg, *.jpeg)|*.jpg;*.jpeg|TGA Files (*.tga)|*.tga",
                                     wxDefaultPosition, wxDefaultSize, wxFLP_USE_TEXTCTRL | wxFLP_OPEN | wxFLP_SMALL);
    fileTextureDiffuse->Bind(wxEVT_FILEPICKER_CHANGED, &UiPanelToolsInput::onFileTextureDiffuse, this);
    sizer->Add(fileTextureDiffuse, wxSizerFlags().Expand().Border(wxDOWN | wxLEFT | wxRIGHT));

    SetSizerAndFit(sizer);
}

void UiPanelToolsInput::refreshProject() {
    fileModel->SetPath(getProject().pathModel);
    fileTextureDiffuse->SetPath(getProject().pathTextureDiffuse);
    frame.rtx->loadModel(getProject().pathModel, [](){});
    frame.rtx->loadTextureDiffuse(getProject().pathTextureDiffuse);
}

void UiPanelToolsInput::onFileModel(wxFileDirPickerEvent& event) {
    // Count the number of lines in the file so we can get a progress estimate
    std::ifstream fileLines(event.GetPath().ToStdString());
    int linesCount = (int)std::count(std::istreambuf_iterator<char>(fileLines), std::istreambuf_iterator<char>(), '\n');

    wxProgressDialog dialog("Loading Model", "Loading model geometry from \"" + event.GetPath().ToStdString() + "\"...", linesCount,
                            frame.panelInput, wxPD_AUTO_HIDE);

    int progress = 0;

    getProject().pathModel = event.GetPath();
    frame.rtx->loadModel(getProject().pathModel, [&](){ dialog.Update(++progress); });
}

void UiPanelToolsInput::onFileTextureDiffuse(wxFileDirPickerEvent& event) {
    getProject().pathTextureDiffuse = event.GetPath();
    frame.rtx->loadTextureDiffuse(getProject().pathTextureDiffuse);
}
