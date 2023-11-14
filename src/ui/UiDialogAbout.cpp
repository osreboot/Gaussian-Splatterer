#include "UiDialogAbout.h"
#include "wx/html/htmlwin.h"

UiDialogAbout::UiDialogAbout(wxWindow* parent) : wxDialog(parent, wxID_ANY, "About Gaussian Splatterer") {
    wxBoxSizer* sizer = new wxBoxSizer(wxVERTICAL);
    this->SetSizer(sizer);

    std::string htmlPage = "<body>"
                           "<p style='text-align: center'>This tool was created by Calvin Weaver (osreboot). The code is available at:</p>"
                           "<p style='text-align: center'>https://github.com/osreboot/Gaussian-Splatterer</p>"
                           "<br/>&nbsp;"
                           "<br/>&nbsp;"
                           "<p style='text-align: center'>Based on research by Bernhard Kerbl, Georgios Kopanas, Thomas Leimk&uuml;hler, and George Drettakis."
                           " This tool also directly invokes their rasterizer. You can find the original paper and more at:</p>"
                           "<p style='text-align: center'>https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/</p>"
                           "<br/>&nbsp;"
                           "<br/>&nbsp;"
                           "<p style='text-align: center'>Other project dependencies:</p>"
                           "<p style='text-align: center'>OptiX Wrapper Library | https://github.com/owl-project/owl</p>"
                           "<p style='text-align: center'>wxWidgets | https://github.com/wxWidgets/wxWidgets</p>"
                           "</body>";

    wxHtmlWindow* html = new wxHtmlWindow(this);
    html->SetPage(htmlPage);
    html->SetMinSize({512, 328});
    sizer->Add(html, wxSizerFlags(1).Border());

    wxButton* buttonClose = new wxButton(this, wxID_ANY, "Ok");
    buttonClose->Bind(wxEVT_COMMAND_BUTTON_CLICKED, &UiDialogAbout::onButtonClose, this);
    sizer->Add(buttonClose, wxSizerFlags().Border().Align(wxCENTER));

    sizer->Fit(this);
    this->Fit();

    Center();
    ShowModal();
    Destroy();
}

void UiDialogAbout::onButtonClose(wxCommandEvent& event) {
    Destroy();
}
