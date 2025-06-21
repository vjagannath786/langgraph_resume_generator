import requests
from bs4 import BeautifulSoup

def get_linkedin_job_description(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("Failed to fetch page:", response.status_code)
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    # LinkedIn's JD is often in a <div> with class 'description__text' or similar
    jd_div = soup.find("div", class_="description__text")
    if jd_div:
        return jd_div.get_text(separator="\n", strip=True)
    else:
        print("Job description not found.")
        return None

if __name__ == "__main__":
    url = "https://www.linkedin.com/jobs/view/4115675695/"
    jd = get_linkedin_job_description(url)
    if jd:
        print("Job Description:")
        print(jd)