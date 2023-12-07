#pragma once

#include <wx/wx.h>
#include <wx/filepicker.h>

#include "Project.h"
#include "ui/UiFrame.h"

class UiPanelToolsInput : public wxPanel {

private:
    UiFrame& frame;
    Project& getProject() const;

    wxFilePickerCtrl* fileModel;
    wxFilePickerCtrl* fileTextureDiffuse;

public:
    UiPanelToolsInput(wxWindow* parent, UiFrame& frame);

    void refreshProject();

    void onFileModel(wxFileDirPickerEvent& event);
    void onFileTextureDiffuse(wxFileDirPickerEvent& event);

};
