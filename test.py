from main import train_classifier, predict_spam, spam_probability, save_classifier, load_classifier

classifier=train_classifier()

#TODO restore proper testing set from data

test_emails = [
    # spam
    "Do you like Sexy Animals doing the wild thing? We have the super hot content on the Internet!\
    This is the site you have heard about. Rated the number one adult site three years in a row!\
    - Thousands of pics from hardcore fucking, and cum shots to pet on girl.\
    - Thousands videos\
    So what are you waiting for?\
     HYPERLINK CLICK HERE\
    YOU MUST BE AT LEAST 18 TO ENTER!\
    You have received this advertisement because you have opted in\
    to receive\
    free adult internet offers and\
    specials through our affiliated websites. If you do not wish to receive\
    further emails or have received the\
    email in error you may opt-out of our database by clicking here:\
     HYPERLINK CLICK HERE\
    Please allow 24hours for removal.\
    This e-mail is sent in compliance with the Information Exchange Promotion and\
    Privacy Protection Act.\
    section 50 marked as 'Advertisement' with valid 'removal' instruction.",
    # spam
    "CNA and CNA Life are registered service marks, trade names and domain\
    names and CNA Maximizer is a registered service mark of CNA Financial\
    Corporation. CNA life insurance products are underwritten by Valley\
    Forge Life Insurance Company, one of the CNA companies. Policy form\
    numbers: V100-1201-A, V100-1202-A, V100-1203-A, V100-1204-A Series.\
    These products and features are not available in all states. For\
    Producer use only.\
    We don't want anybody to receive our mailing who does not wish to\
    receive them. This is a professional communication sent to insurance\
    professionals. To be removed from this mailing list, DO NOT REPLY to\
    this message. Instead, go here: http://www.insuranceiq.com/optout\
    <http://www.insuranceiq.com/optout/> \
    Legal Notice <http://www.insuranceiq.com/legal.htm>",
    # spam
    "Removal instructions below I saw your listing on the internet.  I work for a company that submits websites to search\
    engines. We can submit your website to over 350 of the worlds best search engines and directories for a one time charge of \
    only $39.95. If you would like to put your website in the fast lane and receive more\
    traffic call me on our toll-free number listed below.\
    All work is verified!\
    Sincerely,\
    Brian Franks\
    888-532-8842\
    To be removed call:888-800-6339 Ext. 1377",
    # spam
    "We guarantee you signups before you ever pay a penny! We will show you the green before you \
    ever take out your wallet.  Sign up for FREE and test drive our system.  No Obligation whatsoever No Time \
    Limit on the test drive. Our system is so powerful that the system enrolled over 400 people \
    into my downline the first week. To get signed up for FREE and take a test drive use\
    the link: mailto:workinathome@btamail.net.cn?subject=more_MOSS4_info_please\
    Be sure to request info if the subject line does not! The national attention drawn by this program \
    will drive this program with incredible momentum! Don't wait, if you wait, the next 400 people will \
    be above you. Take your FREE test drive and have the next 400 below you!\
    mailto:workinathome@btamail.net.cn?subject=more_MOSS4_info_please\
    Be sure to request info if the subject line does not! All the best, Daniel Financially Independent Home Business Owner",
    # spam
    "viagra nigerian prince sells SEO cheap enter your pin mobutu widow credit card money send us low rate risk of cancer free",
    # non-spam
    "Help wanted.  We are a 14 year old fortune 500 company, that is\
    growing at a tremendous rate.  We are looking for individuals who\
    want to work from home.\
    This is an opportunity to make an excellent income.  No experience\
    is required.  We will train you.\
    So if you are looking to be employed from home with a career that has\
    vast opportunities, then go:\
    http://www.basetel.com/wealthnow\
    We are looking for energetic and self motivated people.  If that is you\
    than click on the link and fill out the form, and one of our\
    employement specialist will contact you.\
    To be removed from our link simple go to:\
    http://www.basetel.com/remove.html\
    ",
    # non-spam
    "US Airways scored a $500 million bankruptcy court bailout.\
    American Airlines announced a re-org. United Airlines' parent\
    company's stock sank on speculation that UAL is next in line for\
    Chapter 11. It's a wonder none of this happened last September,\
    but it's not much consolation to airline employees,\
    shareholders, and customers that it's happening now. The Wall Street Journal had the most extensive coverage of\
    American's plan as of Unspun's deadline. The airline will lay\
    off 7,000 workers, ground some jets, cut back on flight\
    amenities such as food, and retool its hub operations to make\
    for longer layovers. 'In the past, airlines were loath to have\
    longer connection times because flights were listed in travel\
    agents' computers by elapsed time,' said the Journal. 'Now,\
    online booking tools and search engines most often list flights\
    by price, not time.' In this age of penny-pinching, American's\
    worried about low-cost carriers, not the big guys like, um, US\
    Air and United.",
    # non-spam
    "Dear SoE Community,\
    Forgive the spam, but I wanted to make sure that everyone is aware that\
    we are recruiting to replace the position vacated by Gary Moro.\
    This is a technical position with supervisor responsibilities. The focus is\
    on Solaris and Linux systems administration with Veritas and Irix experience\
    a plus.\
    The posting is here:\
    http://www2.ucsc.edu/staff_hr/employment/listings/020809.htm\
    Regards,Michael Perrone",
    #non-spam
    "> I've got some of the way, but like a similar post earlier about modem\
    > problems, when I am connected to the internet with eht0 up, the routing\
    > is all incorrect and noting goes out through ppp0 (eh0 must be the\
    > default route or something).\
    > Is there standard 'out of the box' Linux tools that will carry out\
    > portmapping on behalf of the LAN PCs ?  (I'm planning on non routable\
    > addresses 192.168.x.x for the LAN, routed outwards via the ppp0\
    > interface). Can someone point me at the right HOWTOs or routing documentation I need\
    > to follow Thanks,Dermot."
]

# In-memory
for email in test_emails:
    res = spam_probability(classifier, email)
    # res = predict_spam(classifier, email)
    # print("The email is non-spam with probability:", p)
    print("Non-spam vs spam probability:", res)

# Saved and loaded
# save_classifier(classifier)
# for email in test_emails:
#     p = spam_probability(load_classifier(), email)
#     print("The email is non-spam with probability:", p)
