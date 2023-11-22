#pragma once

#include <wx/wx.h>
#include <wx/filepicker.h>

#include "Project.h"
#include "ui/UiFrame.h"

class UiPanelToolsInput : public wxPanel {

private:
    UiFrame& getFrame() const;
    Project& getProject() const;

    wxStaticBoxSizer* sizer;

    wxFilePickerCtrl* fileModel;
    wxFilePickerCtrl* fileTextureDiffuse;

public:
    UiPanelToolsInput(wxWindow* parent);

    void refreshProject();

    void onFileModel(wxFileDirPickerEvent& event);
    void onFileTextureDiffuse(wxFileDirPickerEvent& event);

};
